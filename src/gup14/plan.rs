use super::bitset::VertexBitSet;
use super::cand_space::*;
use super::embedding::*;
use super::filtering::*;
use super::gup::*;
use super::reservation::*;
use super::*;
use crate::graph::*;
use crate::gup14::nogood::*;
use crate::utils::*;
use itertools::Itertools;
use log::trace;
use rayon::ThreadPool;

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;

const DUMP_CS: bool = false;

////////////////////////////////////////////////////////////////////////////////
//
// ThreadState
//
////////////////////////////////////////////////////////////////////////////////

/// Mutable data maintained by each thread throughout query processing.
pub struct ThreadState {
    /// `u -> #assignment changes`
    pub assignment_ages: Vec<usize>,
    // Counters for performance analyses
    pub recursion_count: usize,
    pub futile_recursion_count: usize,
    pub guard_count_reservation: usize,
    pub guard_count_vertex_nogood: usize,
    pub guard_count_edge_nogood: usize,
    pub backjump_count: usize,
    pub running_secs: f64,
    pub _task_count: usize,
}

impl ThreadState {
    pub fn new(cs: &CandSpace) -> Self {
        ThreadState {
            // Reserve age 0 as an invalid value
            assignment_ages: vec![1; query_vertex_count(cs)],
            recursion_count: 0,
            futile_recursion_count: 0,
            guard_count_reservation: 0,
            guard_count_vertex_nogood: 0,
            guard_count_edge_nogood: 0,
            backjump_count: 0,
            running_secs: 0.0,
            _task_count: 0,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// SearchEnv
//
////////////////////////////////////////////////////////////////////////////////

/// Set of immutable data for a given query graph
pub struct SearchEnv<'a, F> {
    pub opt: &'a SearchOptions,
    pub pool: &'a ThreadPool,
    pub cs: CandSpace,
    /// Number of core vertices.
    /// Reordered query vertex `u` is in the 2-core iff `u < core_size`.
    pub core_size: usize,
    /// Ordering: (reordered u) -> (original u)
    pub ord: Vec<Vertex>,
    /// Permutation: (original u) -> (reordered u)
    pub perm: Vec<Vertex>,
    /// `u -> [preceding parent of every neighbor of u]`
    /// Preceding parent := `argmax_{u" in N(u')} u" < u`.
    pub preceding_parents: Vec<Vec<Option<Vertex>>>,
    /// Timeout timer
    pub timer: Timer,
    pub reservations: ReservationGuards,
    /// Edges in a query graph that connect query vertices whose IDs differ more
    /// than 2, i.e., (u, u_) s.t. u + 1 < u_.
    pub skipedges: Vec<Edge>,
    pub skipedge_owners: Vec<Vec<usize>>,
    pub skipedge_assignments: Vec<Vec<usize>>,
    /// Callback invoked when a full embedding is found.
    /// `F: Fn(&ReorderedEmbedding) -> bool`
    pub callback: F,
    pub threads: Vec<Mutex<ThreadState>>,
    pub escape: AtomicBool,
    /// Total number of embeddings found in each threads.
    pub match_count: AtomicUsize,
    /// Number of non-idle threads.
    pub running_count: AtomicUsize,
    pub root_cands: Vec<usize>,
}

////////////////////////////////////////////////////////////////////////////////
//
// Public Free Functions
//
////////////////////////////////////////////////////////////////////////////////

/// Returns `None` if it is found that there are no embeddings.
pub fn plan<'a, Q: VertexBitSet, F>(
    ie: &'a Gup,
    q: &Graph,
    callback: F,
) -> Option<SearchEnv<'a, F>>
where
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    let g = ie.g;
    let timer = Timer::started(ie.opt.timeout);

    // Pruning
    let core = bz_kcore(q);
    let (uroot, matchas, hints) = prune::<Q>(q, &core, g, ie);

    // Build a candidate space
    let cs = construct_cs(q, g, &ie.elist, &matchas, &hints);
    if cs[uroot as usize].candidates.len() == 0 {
        return None; // No embedding if `uroot` has no candidate vertex
    }

    // Ordering
    let ord = ordering::optimize_order(q, &cs, uroot, &core);
    let perm = ordering::to_permutation(&ord);
    let cs = cand_space::reorder(cs, &ord, &perm);
    let core_size = core.iter().filter(|&&c| c >= 2).count();
    debug_assert!(ord.iter().take(core_size).all(|&u| core[u as usize] >= 2));
    debug_assert!(ord.iter().skip(core_size).all(|&u| core[u as usize] < 2));

    if DUMP_CS {
        print_details(q, uroot, &cs);
    }

    // Generate reservation guards
    let resvs = ReservationGuards::generate(
        &cs,
        &matchas,
        &ord,
        ie.opt.reservation_size,
    );

    let preceding_parents = compute_preceding_parents(&cs);
    let (skipedges, skipedge_owners, skipedge_assignments) =
        assign_skipedges(&cs, core_size);
    let threads = (0..ie.opt.parallelism)
        .map(|_| Mutex::new(ThreadState::new(&cs)))
        .collect_vec();
    let root_cands = (0..cand_count(&cs, 0)).collect();

    Some(SearchEnv {
        opt: &ie.opt,
        pool: &ie.pool,
        cs,
        core_size,
        ord,
        perm,
        preceding_parents,
        timer,
        reservations: resvs,
        skipedges,
        skipedge_owners,
        skipedge_assignments,
        callback,
        threads,
        escape: AtomicBool::new(false),
        match_count: AtomicUsize::new(0),
        running_count: AtomicUsize::new(0),
        root_cands,
    })
}

pub fn show_probe<Q: VertexBitSet, F>(e: &SearchEnv<F>) {
    let ord_str = e
        .ord
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let gcs_vertex_count =
        e.cs.iter().map(|x| x.candidates.len()).sum::<usize>();
    let gcs_edge_count =
        e.cs.iter()
            .map(|x| {
                x.neighbors
                    .iter()
                    .map(|(_, ivs)| ivs.data_len())
                    .sum::<usize>()
            })
            .sum::<usize>();
    let reservation_total = e.reservations.total_size();
    let reservation_max = e.reservations.max_guard_size();
    let total_bytes_reservation = e.reservations.heap_size_of();

    let mut recursion_count = 0;
    let mut futile_recursion_count = 0;
    let mut guard_count_reservation = 0;
    let mut guard_count_vertex_nogood = 0;
    let mut guard_count_edge_nogood = 0;
    let mut backjump_count = 0;
    for t in &e.threads {
        let t = t.lock().unwrap();
        recursion_count += t.recursion_count;
        futile_recursion_count += t.futile_recursion_count;
        guard_count_reservation += t.guard_count_reservation;
        guard_count_vertex_nogood += t.guard_count_vertex_nogood;
        guard_count_edge_nogood += t.guard_count_edge_nogood;
        backjump_count += t.backjump_count;
    }

    let total_bytes_vertex_nogood =
        allocate_vertex_nogoods::<Q>(&e.cs).heap_size_of() * e.threads.len();
    let total_bytes_edge_nogood =
        allocate_edge_nogoods::<Q>(&e.cs, e.core_size).heap_size_of()
            * e.threads.len();

    println!(
        "  probe:
    matching_order: [{ord_str}]
    gcs_vertex_count: {gcs_vertex_count}
    gcs_edge_count: {gcs_edge_count}
    reservation_total: {reservation_total}
    reservation_max: {reservation_max}
    recursion_count: {recursion_count}
    futile_recursion_count: {futile_recursion_count}
    guard_count_reservation: {guard_count_reservation}
    guard_count_vertex_nogood: {guard_count_vertex_nogood}
    guard_count_edge_nogood: {guard_count_edge_nogood}
    backjump_count: {backjump_count}
    total_bytes_reservation: {total_bytes_reservation}
    total_bytes_vertex_nogood: {total_bytes_vertex_nogood}
    total_bytes_edge_nogood: {total_bytes_edge_nogood}"
    );

    println!("    threads:");
    for t in &e.threads {
        let t = t.lock().unwrap();
        println!("    - running_secs: {}", t.running_secs);
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Private Free Functions
//
////////////////////////////////////////////////////////////////////////////////

fn assign_skipedges(
    cs: &CandSpace,
    core_size: usize,
) -> (Vec<Edge>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut edges = vec![];
    let mut owners = vec![vec![]; query_vertex_count(cs)];
    for u in 0..(core_size as Vertex) {
        for uc in query_neighbors(cs, u) {
            if u + 1 < uc && uc < core_size as Vertex {
                let i = edges.len();
                edges.push((u, uc));
                owners[u as usize].push(i);
                trace!("skipedge: i = {i}, edge = ({u}, {uc})");
            }
        }
    }
    debug_assert!(edges.iter().all_unique());

    let mut assignments = vec![vec![]; query_vertex_count(cs)];
    for i in 0..edges.len() {
        let (u, uc) = edges[i];
        for um in u + 1..uc {
            assignments[um as usize].push(i);
        }
    }

    (edges, owners, assignments)
}

fn compute_preceding_parents(cs: &CandSpace) -> Vec<Vec<Option<Vertex>>> {
    query_vertices(cs)
        .map(|u| {
            query_neighbors(cs, u)
                .map(|u_| {
                    if u_ < u {
                        // A preceding parent is not defined for a parent
                        None
                    } else {
                        query_neighbors(cs, u_)
                            .take_while(|&ucp| ucp < u)
                            .last()
                    }
                })
                .collect()
        })
        .collect()
}
