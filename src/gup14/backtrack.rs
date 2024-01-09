use super::bitset::*;
use super::cand_space::*;
use super::embedding::*;
use super::nogood::*;
use super::plan::*;
use crate::graph::*;
use crate::utils::*;
use itertools::{EitherOrBoth, Itertools};
use log::trace;
use rayon::Scope;
use std::cmp::Ord;

use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Instant;

////////////////////////////////////////////////////////////////////////////////
//
// BacktrackArgs
//
////////////////////////////////////////////////////////////////////////////////

/// Arguments to start a new backtracking search.
///
/// `'a` is a lifetime of `SearchEnv<F>`.
pub struct BacktrackArgs<'a, Q: VertexBitSet> {
    m: Embedding,
    cand_stack: Vec<Vec<Vec<LocalCand<Q>>>>,
    tree_cands: Vec<&'a [usize]>,
    skipedge_masks: Vec<Q>,
    boundings: Vec<Q>,
}

impl<'a, Q: VertexBitSet> BacktrackArgs<'a, Q> {
    /// Returns args for the initial backtracking search
    pub fn origin<F>(e: &'a SearchEnv<F>) -> Self {
        let mut cand_stack = vec![vec![]; e.core_size];
        let mut tree_cands = vec![&[] as &[usize]; query_vertex_count(&e.cs)];

        if e.core_size > 0 {
            cand_stack[0] = vec![(0..cand_count(&e.cs, 0))
                .map(|cand_index| {
                    LocalCand::new(cand_index, usize::MAX, usize::MAX)
                })
                .collect()];
        } else {
            tree_cands[0] = e.root_cands.as_slice();
        }

        let mut boundings = vec![VertexBitSet::ZERO; query_vertex_count(&e.cs)];
        for u in e.core_size.max(1)..query_vertex_count(&e.cs) {
            // Bounding sets for tree vertices are never changed
            let parent = query_neighbors(&e.cs, u as Vertex).next().unwrap();
            boundings[u] = Q::from_vertex(parent);
        }

        BacktrackArgs {
            m: Embedding::new(query_vertex_count(&e.cs)),
            cand_stack,
            tree_cands,
            skipedge_masks: vec![VertexBitSet::ZERO; e.skipedges.len()],
            boundings,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// BacktrackState
//
////////////////////////////////////////////////////////////////////////////////

/// Mutable thread-local state for a single backtracking search
pub struct BacktrackState<'a, 'b: 'a, Q: VertexBitSet, F> {
    env: &'b SearchEnv<'b, F>,
    u_bottom: Vertex,
    thread: &'a mut ThreadState,
    scope: &'a Scope<'b>,
    m: Embedding,
    /// `u -> stack of [LocalCand]`. `u` must be a core vertex.
    cand_stack: Vec<Vec<Vec<LocalCand<Q>>>>,
    /// `u -> [cand indices]`. `u` must be a tree vertex.
    /// `tree_cands[0..env.core_size]` is not used.
    tree_cands: Vec<&'b [usize]>,
    /// u -> position of the current local candidate vertex in `cand_stack`
    local_cand_indices: Vec<usize>,
    skipedge_masks: Vec<Q>,
    /// u -> bounding set
    boundings: Vec<Q>,
    /// Backed-up data used for revering the result of refinement
    refinement_backups: Vec<Q>,
    /// Copy of `SearchEnv::match_count` at the last sync
    match_count_cache: usize,
    /// Number of embeddings found in this thread after the last sync
    match_increment: usize,
    /// `u -> iv -> Nogood`
    vertex_nogoods: Vec<Vec<Nogood<Q>>>,
    /// Similar structure to `CandidateSpace::neighbors`;
    /// `u -> [(u', iv -> [edge nogood for each adjacent candidate])]`
    edge_nogoods: Vec<Vec<(Vertex, Vec<Vec<Nogood<Q>>>)>>,
}

impl<'a, 'b: 'a, Q: VertexBitSet, F> BacktrackState<'a, 'b, Q, F> {
    fn cand_count(&self, u: Vertex) -> usize {
        if u < self.env.core_size as Vertex {
            self.cand_stack[u as usize]
                .last()
                .map(|x| x.len())
                .unwrap_or(0)
        } else {
            self.tree_cands[u as usize].len()
        }
    }

    /// Returns the candidate index of the `index`-th current local candidate of
    /// `u`.
    fn local_cand(&self, u: Vertex, index: usize) -> usize {
        if u < self.env.core_size as Vertex {
            self.cand_stack[u as usize].last().unwrap()[index].cand_index
        } else {
            self.tree_cands[u as usize][index]
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// BacktrackCall
//
////////////////////////////////////////////////////////////////////////////////

#[allow(dead_code)]
struct BacktrackCall<'a, 'b, Q: VertexBitSet, F> {
    u: Vertex,
    /// Union of the conflict masks.
    conflict_mask: Q,
    /// Union of the deadend masks. `Ok(())` if at least one full embedding is
    /// found in and `Err(k)` if not.
    deadend_mask: Result<(), Q>,
    /// Index in `refinment_backups` where backup data of this call are stored
    backup_origin_index: usize,
    _phantom: PhantomData<(&'a (), &'b (), F)>,
}

#[allow(dead_code)]
enum Next<'a, 'b, Q: VertexBitSet, F> {
    Recurse(BacktrackCall<'a, 'b, Q, F>),
    Return(Result<(), Q>),
    Continue(Result<(), Q>),
}

#[allow(dead_code)]
impl<'a, 'b, Q: VertexBitSet, F> BacktrackCall<'a, 'b, Q, F>
where
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    fn new(s: &BacktrackState<'a, 'b, Q, F>, u: Vertex) -> Self {
        BacktrackCall {
            u,
            conflict_mask: Q::ZERO,
            // `Err`: any full embeddings have not yet been found so far.
            // `!Q::ZERO`: expect to be replaced by rule (3) of Def. 3.27.
            deadend_mask: Err(!Q::ZERO),
            backup_origin_index: s.refinement_backups.len(),
            _phantom: PhantomData,
        }
    }

    fn next(
        &mut self,
        s: &mut BacktrackState<'a, 'b, Q, F>,
        prev_result: Option<Result<(), Q>>,
    ) -> Next<'a, 'b, Q, F> {
        let u = self.u;

        if let Some(x) = prev_result {
            self.accumulate(s, x);

            // To let `local_cand_indices` point the current local candidate
            // vertex during the subsequent recursions, we increment the
            // index after the loop body
            s.local_cand_indices[u as usize] += 1;
        } else {
            // `prev_result == None` means that this is the first iteration
            s.local_cand_indices[u as usize] = 0;
        }

        debug_assert!(s.cand_count(u) > 0);
        if s.local_cand_indices[u as usize] >= s.cand_count(u) {
            return Next::Return(self.finalize(s));
        }

        let iv = s.local_cand(u, s.local_cand_indices[u as usize]);
        match extend(s, u, iv) {
            Err(k) => {
                self.conflict_mask |= k;
                Next::Continue(Err(k))
            }
            Ok(()) => match try_recurse(s) {
                None => Next::Continue(Ok(())),
                Some(c) => Next::Recurse(c),
            },
        }
    }

    fn accumulate(
        &mut self,
        s: &mut BacktrackState<'a, 'b, Q, F>,
        x: Result<(), Q>,
    ) {
        // If the previous extension was successful...
        if s.m.len() > self.u as usize {
            debug_assert_eq!(s.m.len(), self.u as usize + 1);
            let iv =
                s.local_cand(self.u, s.local_cand_indices[self.u as usize]);
            update_edge_nogoods(s, self.u, iv);
            unextend(s, self.u, self.backup_origin_index);
        }
        update_nogood(
            s,
            self.u,
            &x,
            &mut self.conflict_mask,
            &mut self.deadend_mask,
        );
    }

    fn finalize(
        &mut self,
        s: &mut BacktrackState<'a, 'b, Q, F>,
    ) -> Result<(), Q> {
        let u = self.u;

        accumulate_to_skipedges(s, u, &self.conflict_mask);

        if let Err(mut k) = self.deadend_mask {
            debug_assert!(k.last_one() <= Some(u as usize));
            if k[u as usize] {
                k.set(u as usize, false);
                k |= s.boundings[u as usize]
            }

            s.thread.futile_recursion_count += 1;
            Err(k)
        } else {
            Ok(())
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// LocalCand
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug)]
struct LocalCand<Q: VertexBitSet> {
    cand_index: usize,
    neighbor_index: usize,
    prev_local_index: usize,
    mask: Result<(), Q>,
}

impl<Q: VertexBitSet> LocalCand<Q> {
    fn new(
        cand_index: usize,
        neighbor_index: usize,
        prev_local_index: usize,
    ) -> Self {
        LocalCand {
            cand_index,
            neighbor_index,
            prev_local_index,
            // The result is initialized with `Err(!Q::ZERO)`.
            // This will be replaced at the first merge because it must not
            // contain the last query vertex
            mask: Err(!Q::ZERO),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// CandStackDisplay
//
////////////////////////////////////////////////////////////////////////////////

struct CandStackDisplay<'a, Q: VertexBitSet>(
    &'a Vec<Vec<Vec<LocalCand<Q>>>>,
    &'a CandSpace,
);

impl<'a, Q: VertexBitSet> fmt::Display for CandStackDisplay<'a, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let candstack = self.0;
        let cs = self.1;

        // Build temporary `Vec` for simplicity though it is inefficient
        let vec = (0..candstack.len())
            .map(|u| {
                candstack[u]
                    .iter()
                    .map(|u_stack| {
                        u_stack
                            .iter()
                            .map(|lc| cands(cs, u as Vertex)[lc.cand_index])
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        write!(f, "{vec:?}")
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Public Free Functions
//
////////////////////////////////////////////////////////////////////////////////

pub fn backtrack_parallel<Q, F>(e: &SearchEnv<F>)
where
    Q: VertexBitSet,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    e.running_count.fetch_add(1, SeqCst); // Count the initial thread
    e.pool.install(|| {
        rayon::scope(|sc| {
            sc.spawn(|sc| {
                backtrack(e, sc, BacktrackArgs::<Q>::origin(e));
            })
        })
    });
}

////////////////////////////////////////////////////////////////////////////////
//
// Private Free Functions
//
////////////////////////////////////////////////////////////////////////////////

fn lift<Q>(mut mask: Q, target_depth: Vertex, bounding_sets: &[Q]) -> Q
where
    Q: VertexBitSet,
{
    while let Some(u) = last_vertex(&mask) {
        if u <= target_depth {
            break;
        }
        mask |= bounding_sets[u as usize];
        mask.set(u as usize, false);
    }

    mask
}

/// Recursively apply rule (4) of Def. 3.27 (Deadend mask) in the paper to
/// obtain the deadend mask of M[:target_depth + 1]
fn add_lift<Q>(mut mask: Q, target_depth: Vertex, bounding_sets: &[Q]) -> Q
where
    Q: VertexBitSet,
{
    let mut ret = mask;

    while let Some(u) = last_vertex(&mask) {
        if u <= target_depth {
            break;
        }
        ret |= bounding_sets[u as usize];
        mask |= bounding_sets[u as usize];
        mask.set(u as usize, false);
    }

    ret
}

fn accumulate_mask<Q>(mask: &Q, accum: &mut Result<(), Q>)
where
    Q: VertexBitSet,
{
    match accum {
        // Replace the mask based on rule (3) of Def.3.27 (Deadend mask)
        Err(k) if mask.last_one() < k.last_one() => *k = *mask,
        // Compute the union to obtain a deadend mask by applying rule (4) later
        Err(k) if mask.last_one() == k.last_one() => *k |= *mask,
        // Unchanged if `accum` is `Ok(())` or `mask.last_one() > k.last_one()`
        _ => (),
    }
}

fn check_injectivity_conflict<Q, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
) -> Result<(), Q>
where
    Q: VertexBitSet,
{
    if let Some(u_) = s.m.invert(cands(&s.env.cs, u)[iv]) {
        Err(Q::from_vertices([u_, u]))
    } else {
        Ok(())
    }
}

fn check_reservation_conflict<Q, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
) -> Result<(), Q>
where
    Q: VertexBitSet,
{
    // If the size limit of a reservation guard is set to zero, this check has
    // no effect because all the guards are `None`.
    let r = s.env.reservations.check(&s.m, u, iv);
    if r.is_err() {
        trace!("reservation_contained: {{ m: {} }}", &s.m,);
        s.thread.guard_count_reservation += 1;
    }
    r
}

fn check_vertex_nogood_conflict<Q, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
) -> Result<(), Q>
where
    Q: VertexBitSet,
{
    if s.env.opt.no_vertex_nogood {
        return Ok(());
    }

    let n = &s.vertex_nogoods[u as usize][iv];
    let r = n.check(&s.thread.assignment_ages, u);
    if r.is_err() {
        trace!(
            "prune_vertex: ({u}, {v}) by {n}",
            v = cands(&s.env.cs, u)[iv],
        );
        s.thread.guard_count_vertex_nogood += 1;
    }
    r
}

fn refine_core_neighbor<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
    iadj: usize,
) -> (Vec<LocalCand<Q>>, Q) {
    let e = s.env;
    let uc = e.cs[u as usize].neighbors[iadj].0;
    // Indices of uc's candidates adjacent to (u, iv)
    let inbs = &e.cs[u as usize].neighbors[iadj].1[iv];
    let nogoods = &s.edge_nogoods[u as usize][iadj].1[iv];

    let mut check = |neighbor_index: usize| {
        let n = &nogoods[neighbor_index];
        match n.check(&s.thread.assignment_ages, u) {
            Err(k) if !e.opt.no_edge_nogood => {
                trace!(
                    "prune_edge: (({u}, {v}), ({uc}, {vc})) by {n}",
                    v = e.cs[u as usize].candidates[iv],
                    vc = e.cs[uc as usize].candidates[inbs[neighbor_index]],
                );
                s.thread.guard_count_edge_nogood += 1;
                Err(k)
            }
            _ => Ok(()),
        }
    };

    if s.cand_stack[uc as usize].is_empty() {
        debug_assert!(s.boundings[uc as usize].not_any());
        let mut lcs = Vec::with_capacity(inbs.len());
        let mut bset = Q::from_vertex(u);
        for (inbr, &ivc) in inbs.iter().enumerate() {
            if let Err(k) = check(inbr) {
                bset |= k;
            } else {
                lcs.push(LocalCand::new(ivc, inbr, usize::MAX));
            }
        }
        return (lcs, bset);
    }

    let ucp = e.preceding_parents[u as usize][iadj].unwrap();
    let current_lcs = s.cand_stack[uc as usize].last_mut().unwrap();
    let mut lcs = Vec::with_capacity(current_lcs.len());
    let mut bset = s.boundings[uc as usize];
    // Precompute because this value is used many times and `lift` is expensive
    let lifted_u = add_lift(Q::from_vertex(u), ucp, &s.boundings);

    for e in itertools::merge_join_by(
        current_lcs.iter_mut().enumerate(),
        inbs.iter().cloned().enumerate(),
        |(_, lc), (_, icand)| lc.cand_index.cmp(icand),
    ) {
        match e {
            EitherOrBoth::Both((iploc, lc), (inbr, _)) => {
                if let Err(k) = check(inbr) {
                    bset |= k;
                    accumulate_mask(
                        &add_lift(k, ucp, &s.boundings),
                        &mut lc.mask,
                    );
                } else {
                    lcs.push(LocalCand::new(lc.cand_index, inbr, iploc));
                }
            }
            EitherOrBoth::Left((_, lc)) => {
                bset.set(u as usize, true);
                accumulate_mask(&lifted_u, &mut lc.mask);
            }
            EitherOrBoth::Right(_) => (),
        }
    }

    (lcs, bset)
}

fn refine<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
) -> Result<(), Q> {
    let e = s.env;
    let backup_origin_index = s.refinement_backups.len();

    for (iadj, &(uc, ref inbsof)) in
        e.cs[u as usize].neighbors.iter().enumerate()
    {
        if uc <= u {
            // TODO: this case can be removed by precomputation
        } else if uc < e.core_size as Vertex {
            let (cands, bset) = refine_core_neighbor(s, u, iv, iadj);
            if cands.is_empty() {
                unrefine(s, u, backup_origin_index);
                return Err(bset);
            }
            s.refinement_backups.push(s.boundings[uc as usize]);
            s.cand_stack[uc as usize].push(cands);
            s.boundings[uc as usize] = bset;
        } else {
            debug_assert_eq!(s.boundings[uc as usize], Q::from_vertex(u));
            s.tree_cands[uc as usize] = &inbsof[iv];
        }
    }

    Ok(())
}

fn unrefine<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    backup_origin_index: usize,
) {
    for (uc, b) in query_neighbors(&s.env.cs, u)
        .skip_while(|&uc| uc <= u)
        .zip(s.refinement_backups[backup_origin_index..].iter())
    {
        debug_assert!((uc as usize) < s.env.core_size);
        s.cand_stack[uc as usize].pop();
        s.boundings[uc as usize] = *b;
    }

    s.refinement_backups.truncate(backup_origin_index);

    // Note: `tree_cands` for tree vertices are left dirty
}

/// Extend a partial embedding and returns a conflict mask in case of failure.
fn extend<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
) -> Result<(), Q> {
    debug_assert_eq!(s.m.len(), u as usize);

    // Check lightweight conditions earlier
    check_vertex_nogood_conflict(s, u, iv)?;
    check_injectivity_conflict(s, u, iv)?;
    check_reservation_conflict(s, u, iv)?;
    refine(s, u, iv)?; // Checks no-candidate conflict

    // Add an assignment after checking the conflicts
    s.m.push(cands(&s.env.cs, u)[iv], iv);
    s.thread.assignment_ages[u as usize] += 1;

    Ok(())
}

fn unextend<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    backup_origin_index: usize,
) {
    unrefine(s, u, backup_origin_index);
    s.m.pop();
}

fn find_skipedge_mask<Q: VertexBitSet, F>(
    s: &BacktrackState<Q, F>,
    u: Vertex,
    uc: Vertex,
) -> Option<Q> {
    debug_assert!(u < uc);
    // TODO: statically compute
    s.env.skipedge_owners[u as usize]
        .iter()
        .find(|&&i| s.env.skipedges[i].1 == uc)
        .map(|&i| s.skipedge_masks[i])
}

fn update_edge_nogoods<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    iv: usize,
) {
    let e = s.env;

    if (u as usize) >= e.core_size {
        // `u` is not in the 2-core
        return;
    }

    // Must be called after extension
    debug_assert_eq!(s.m.len(), u as usize + 1);

    for (iadj, &(uc, _)) in e.cs[u as usize].neighbors.iter().enumerate() {
        if uc <= u {
            continue;
        }
        if e.core_size <= uc as usize {
            break;
        }

        debug_assert_eq!(s.edge_nogoods[u as usize][iadj].0, uc);
        let ucp = e.preceding_parents[u as usize][iadj];

        for ilocal in 0..s.cand_stack[uc as usize].last().unwrap().len() {
            let lc = s.cand_stack[uc as usize].last().unwrap()[ilocal].clone();

            if let Err(mut k) = lc.mask {
                if k.last_one() > Some(u as usize) {
                    if let Some(sk) = find_skipedge_mask(s, u, uc) {
                        k |= sk;
                    }
                }
                k.clear_after(u as usize + 1);

                if let Some(prevcands) =
                    RevIndex(&mut s.cand_stack[uc as usize]).get_mut(1)
                {
                    accumulate_mask(
                        &add_lift(k, ucp.unwrap(), &s.boundings),
                        &mut prevcands[lc.prev_local_index].mask,
                    );
                }

                let n = Nogood::encode(k, u, &s.thread.assignment_ages);
                trace!(
                    "set_edge_nogood: (({u}, {v}), ({uc}, {vc})), {:?}",
                    &n,
                    v = cands(&e.cs, u)[iv],
                    vc = cands(&e.cs, uc)[lc.cand_index],
                );
                s.edge_nogoods[u as usize][iadj].1[iv][lc.neighbor_index] = n;
            } else {
                // `lc.mask` is `Ok(())` iff a full embedding is found.
                // In this case, we can leave the nogood guard on the edge
                // unchanged because it is already ineffective. (If not, `lc`
                // must have been filtered out from the local candidate-vertex
                // set.)

                // Propagate "matched" to the edge between ucp & uc
                if let Some(prevcands) =
                    RevIndex(&mut s.cand_stack[uc as usize]).get_mut(1)
                {
                    prevcands[lc.prev_local_index].mask = Ok(());
                }

                // Sanity check; if an edge between `(u, iv)` and `lc`
                // contributed to find a full embedding, the nogood guard of
                // that edge must not be matched by the current partial
                // embedding.
                // Note that this check is meaningful only if edge-nogood is
                // disabled because `lc` was already filtered out if its guard
                // was matched.
                if e.opt.no_edge_nogood {
                    debug_assert!({
                        let n = &s.edge_nogoods[u as usize][iadj].1[iv]
                            [lc.neighbor_index];
                        n.check(&s.thread.assignment_ages, u).is_ok()
                    });
                }
            }
        }
    }
}

fn format_masked_extension<Q: VertexBitSet>(
    m: &Embedding,
    extension: Vertex,
    r: &Result<(), Q>,
) -> String {
    let k = r.err().unwrap_or(Q::ZERO);
    let body = (0..m.len())
        .map(|u| m.vertex(u as Vertex))
        .chain([extension])
        .enumerate()
        .map(|(u, v)| {
            if k[u] {
                format!("*{}*", v)
            } else {
                v.to_string()
            }
        })
        .join(", ");
    format!("[{body}]")
}

fn update_nogood<Q: VertexBitSet, F>(
    s: &mut BacktrackState<Q, F>,
    u: Vertex,
    r: &Result<(), Q>,
    conflict_mask: &mut Q,
    deadend_mask: &mut Result<(), Q>,
) {
    let e = s.env;
    let iv = s.local_cand(u, s.local_cand_indices[u as usize]);

    debug_assert!(r.is_ok() || r.unwrap_err().last_one() <= Some(u as usize));

    trace!(
        "ext: {:3} rec = {:6}, M = {}",
        if r.is_ok() { "Ok" } else { "Err" },
        s.thread.recursion_count,
        &format_masked_extension(&s.m, e.cs[u as usize].candidates[iv], &r),
    );

    if let Err(k) = *r {
        // Update the vertex nogood of the candidate
        debug_assert!(k.any() && (s.m.len() == u as usize));
        let n = Nogood::encode(k, u, &s.thread.assignment_ages);
        trace!(
            "set_vertex_nogood: (u{u}, v{v}), {n:?}",
            v = e.cs[u as usize].candidates[iv],
        );
        s.vertex_nogoods[u as usize][iv] = n;

        // Accumulate the mask of the extension to the mask of the current
        // embedding
        accumulate_mask(&k, deadend_mask);

        if k[u as usize] || e.opt.no_backjumping {
            if u < e.core_size as Vertex {
                let lcs = s.cand_stack[u as usize].last_mut().unwrap();
                // Accumulate the mask of the extension to the candidate edge
                // Remove endpoint `u` before the accumulation
                accumulate_mask(
                    &(k & !Q::from_vertex(u)),
                    &mut lcs[s.local_cand_indices[u as usize]].mask,
                );
            }
        } else {
            trace!("backjump");
            s.thread.backjump_count += 1;
            if u < e.core_size as Vertex {
                let lcs = s.cand_stack[u as usize].last_mut().unwrap();
                // Accumulate the mask to all the remaining candidates
                // Note: `k & !make_bitset([u])` is not needed because
                //       `k[u] = false`.
                for lc in &mut lcs[s.local_cand_indices[u as usize]..] {
                    accumulate_mask(&k, &mut lc.mask);
                }
            }

            // Skip all; do not use `usize::MAX` because it may cause overflow
            s.local_cand_indices[u as usize] = s.cand_count(u);

            // Nogood guards on skipedges must contain assignments that
            // reproduce a backjump
            *conflict_mask |= k;
        }
    } else {
        *deadend_mask = Ok(());
        if u < e.core_size as Vertex {
            let lcs = s.cand_stack[u as usize].last_mut().unwrap();
            lcs[s.local_cand_indices[u as usize]].mask = Ok(());
        }
    }
}

fn accquire_idle_thread<F>(e: &SearchEnv<F>) -> bool {
    loop {
        let n = e.running_count.load(SeqCst);
        if n >= e.opt.parallelism {
            return false;
        }
        if e.running_count
            .compare_exchange(n, n + 1, SeqCst, SeqCst)
            .is_ok()
        {
            return true;
        }
    }
}

fn spawn_task<'a, 'b: 'a, Q, F>(
    s: &mut BacktrackState<'a, 'b, Q, F>,
    u_spawn: Vertex,
) -> BacktrackArgs<'b, Q>
where
    Q: VertexBitSet,
{
    let e = s.env;

    // to reduce casts
    let u_spawn = u_spawn as usize;
    let u_bottom = s.u_bottom as usize;

    let mut mask = Q::ZERO;
    mask.fill_range(0..u_spawn, true);
    let mut boundings = s.boundings.iter().map(|&b| b & mask).collect_vec();
    let skipedge_masks =
        s.skipedge_masks.iter().map(|&b| b & mask).collect_vec();

    let mut cand_stack = vec![vec![]; e.core_size];
    let mut tree_cands = vec![&[] as &[usize]; query_vertex_count(&e.cs)];

    // For u < u_spawn: Local candidate vertices are not cloned

    // For u = u_spawn
    if u_spawn < e.core_size {
        // Move the half of the local candidates of `u_spawn` from
        // `s.cand_stack` to `cand_stack`
        let from = &mut s.cand_stack[u_spawn].last_mut().unwrap();
        let icurrent = s.local_cand_indices[u_spawn as usize];
        let nrest = from.len() - icurrent;
        debug_assert!(nrest >= 2);
        let isplit = icurrent + nrest / 2;
        cand_stack[u_spawn] = vec![from[isplit..].to_vec()];
        from.truncate(isplit);
    } else {
        let from = s.tree_cands[u_spawn];
        let icurrent = s.local_cand_indices[u_spawn as usize];
        let nrest = from.len() - icurrent;
        debug_assert!(nrest >= 2);
        let isplit = icurrent + nrest / 2;
        tree_cands[u_spawn] = &from[isplit..];
        s.tree_cands[u_spawn] = &from[..isplit];
    }

    // For u s.t. u_spawn < u < core_size
    for u in (u_spawn + 1)..e.core_size {
        // We assume that the query graph is connected and sorted, and hence
        // (i) every query vertex has at least one neighbor (unwrap-able)
        // (ii) the first neighbor has the smallest ID
        let firstnbr = query_neighbors(&e.cs, u as Vertex).next().unwrap();

        // If none of vertices `0..u_spawn` are connected to `u`, `u` does not
        // have local candidate vertices.
        if firstnbr >= u_spawn as Vertex {
            // Since `boundings[u]` may contain a vertex that came from the
            // deadend mask of a nogood guard on edges, it may not be empty even
            // if vertices `0..u_spawn` are not conected to `u`. So here it is
            // cleared.
            boundings[u] = VertexBitSet::ZERO;
            continue;
        }

        // Calculate the index of the stack top at the time when vertices
        // `0..u_spawn` have assignments.
        // If any of vertices `0..u_bottom` is connected to `u`, `cand_stack[u]`
        // had at least one stack level when this thread was spawned.
        let norigin = if firstnbr < u_bottom as Vertex { 1 } else { 0 };
        // Count the number of neighbors within `u_bottom..u_spawn`
        let npushed = query_neighbors(&e.cs, u as Vertex)
            .skip_while(|&up| up < u_bottom as Vertex)
            .take_while(|&up| up < u_spawn as Vertex)
            .count();
        let itop = norigin + npushed - 1; // -1 to get an index
        cand_stack[u] = vec![s.cand_stack[u][itop].clone()];

        // `u` must have a nonempty bounding set if it has local candidates
        debug_assert!(boundings[u].any());
    }

    // For u s.t. max(core_size, u_spawn + 1) <= u
    for u in e.core_size.max(u_spawn + 1)..query_vertex_count(&e.cs) {
        // `s.tree_cands` might be dirty (unused old value), but it is harmless
        // to copy them
        tree_cands[u] = s.tree_cands[u];
        boundings[u] = s.boundings[u];
    }

    BacktrackArgs {
        m: s.m.copy(u_spawn),
        cand_stack,
        tree_cands,
        skipedge_masks,
        boundings,
    }
}

fn try_spawn_task<'a, 'b: 'a, Q, F>(s: &mut BacktrackState<'a, 'b, Q, F>)
where
    Q: VertexBitSet + 'b,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    // Are there any idle threads?
    if s.env.running_count.load(SeqCst) < s.env.opt.parallelism {
        // Can I spawn the search?
        if let Some(u_spawn) = (s.u_bottom as usize..s.m.len())
            .find(|&u| s.cand_count(u as Vertex) - s.local_cand_indices[u] >= 2)
        {
            if accquire_idle_thread(s.env) {
                trace!("New task will be submitted");
                let args = spawn_task(s, u_spawn as Vertex);
                s.scope.spawn(|sc| backtrack(s.env, sc, args));
            }
        }
    }
}

fn synchronize<'a, 'b: 'a, Q, F>(s: &mut BacktrackState<'a, 'b, Q, F>)
where
    Q: VertexBitSet,
{
    let e = s.env;

    s.match_count_cache =
        e.match_count.fetch_add(s.match_increment, SeqCst) + s.match_increment;
    s.match_increment = 0;

    if s.match_count_cache >= e.opt.match_limit || e.timer.is_over() {
        e.escape.store(true, SeqCst);
    }
}

fn accumulate_to_skipedges<'a, 'b: 'a, Q: VertexBitSet, F>(
    s: &mut BacktrackState<'a, 'b, Q, F>,
    u: Vertex,
    conflict_mask: &Q,
) {
    let e = s.env;

    debug_assert!(e.skipedge_assignments[u as usize]
        .iter()
        .map(|&i| e.skipedges[i].0)
        .is_ordered());

    let mut k = *conflict_mask | s.boundings[u as usize];
    for &ie in e.skipedge_assignments[u as usize].iter().rev() {
        let (us, _ut) = e.skipedges[ie];
        // `lift(conflict_mask | s.boundings[u], us, &s.boundings)` can be
        // computed in this manner because `us` monotonically decreases
        k = lift(k, us, &s.boundings);
        // `merge_mask` is not applicable here.
        // Overwriting is allowed only if `conflict_mask` does not contain a
        // query vertex greater than `_ut`, and here we cannot check it.
        s.skipedge_masks[ie] |= k;
    }
}

fn search_recursion_end<'a, 'b: 'a, Q: VertexBitSet, F>(
    s: &mut BacktrackState<'a, 'b, Q, F>,
) -> Result<(), Q>
where
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    debug_assert_eq!(s.m.len(), query_vertex_count(&s.env.cs));

    // Count this as a recursion for consistency with the SIGMOD paper
    s.thread.recursion_count += 1;

    let e = s.env;
    let re = ReorderedEmbedding::new(&s.m, &e.perm);
    let cont = (s.env.callback)(&re);

    s.match_increment += 1;
    let nmatch = s.match_count_cache + s.match_increment;

    if !cont || nmatch >= e.opt.match_limit {
        e.escape.store(true, SeqCst);
    }

    Ok(())
}

#[allow(dead_code)]
fn search_recursion<'a, 'b: 'a, Q, F>(
    s: &mut BacktrackState<'a, 'b, Q, F>,
) -> Result<(), Q>
where
    Q: VertexBitSet + 'b,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    let e = s.env;
    let u = s.m.len() as Vertex;
    // Union of the conflict masks.
    let mut conflict_mask = VertexBitSet::ZERO;
    // Union of the deadend masks. `Ok(())` if at least one full embedding is
    // found in and `Err(k)` if not.
    let mut deadend_mask = Err(!Q::ZERO);
    // Index in `refinment_backups` where backup data of this call are stored
    let backup_origin_index = s.refinement_backups.len();

    debug_assert!(s.cand_count(u) > 0);
    s.local_cand_indices[u as usize] = 0;
    s.thread.recursion_count += 1;

    // Clear skipedges connected to `u_`
    for &ie in &e.skipedge_owners[u as usize] {
        s.skipedge_masks[ie] = VertexBitSet::ZERO;
    }

    trace!("recurse: rec = {:6}, M = {}", s.thread.recursion_count, s.m);

    // `s.cand_count(u)` may change during the loop because of spawns
    while s.local_cand_indices[u as usize] < s.cand_count(u) {
        // `== 1` for spawning at the first iteration
        if s.thread.recursion_count % 0xff == 1 {
            synchronize(s);
            try_spawn_task(s);
        }

        if e.escape.load(SeqCst) {
            // Use `return` for escape to avoid reaching the code after the loop
            // with `deadend_mask = Err(!Q::ZERO)`
            return Ok(());
        }

        let iv = s.local_cand(u, s.local_cand_indices[u as usize]);
        let r = match extend(s, u, iv) {
            Err(k) => {
                conflict_mask |= k;
                Err(k)
            }
            Ok(()) => {
                let r = if u as usize + 1 == query_vertex_count(&e.cs) {
                    search_recursion_end(s) // Expected to be inlined
                } else {
                    search_recursion(s)
                };
                debug_assert_eq!(s.m.len(), u as usize + 1);
                update_edge_nogoods(s, u, iv);
                unextend(s, u, backup_origin_index);
                r
            }
        };

        update_nogood(s, u, &r, &mut conflict_mask, &mut deadend_mask);
        s.local_cand_indices[u as usize] += 1;
    }

    accumulate_to_skipedges(s, u, &conflict_mask);

    if let Err(mut k) = deadend_mask {
        debug_assert!(k.last_one() <= Some(u as usize));
        if k[u as usize] {
            k.set(u as usize, false);
            k |= s.boundings[u as usize]
        }
        s.thread.futile_recursion_count += 1;
        Err(k)
    } else {
        Ok(())
    }
}

#[allow(dead_code)]
fn try_recurse<'a, 'b, Q, F>(
    s: &mut BacktrackState<'a, 'b, Q, F>,
) -> Option<BacktrackCall<'a, 'b, Q, F>>
where
    Q: VertexBitSet,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    let e = s.env;
    let u = s.m.len() as Vertex;

    if u as usize == query_vertex_count(&e.cs) {
        let _ = search_recursion_end(s);
        None
    } else {
        trace!("recurse: rec = {:6}, M = {}", s.thread.recursion_count, s.m);
        // Clear skipedges connected to `u`
        for &ie in &e.skipedge_owners[u as usize] {
            s.skipedge_masks[ie] = VertexBitSet::ZERO;
        }
        s.thread.recursion_count += 1;
        Some(BacktrackCall::new(s, u))
    }
}

#[allow(dead_code)]
fn search_loop<'a, 'b: 'a, Q, F>(s: &mut BacktrackState<'a, 'b, Q, F>)
where
    Q: VertexBitSet + 'b,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    let e = s.env;
    let mut result = None;
    let mut stack = Vec::with_capacity(query_vertex_count(&e.cs));

    stack.push(BacktrackCall::new(s, s.u_bottom));
    s.thread.recursion_count += 1; // Count the first call

    while !stack.is_empty() && !e.escape.load(SeqCst) {
        // Termination check and synchronization with an interval
        if s.thread.recursion_count % 0xff == 1 {
            synchronize(s);
            try_spawn_task(s);
        }

        match stack.last_mut().unwrap().next(s, result) {
            Next::Continue(r) => {
                result = Some(r);
            }
            Next::Recurse(s) => {
                result = None;
                stack.push(s);
            }
            Next::Return(x) => {
                result = Some(x);
                stack.pop();
            }
        }
    }
}

pub fn backtrack<'a, Q, F>(
    e: &'a SearchEnv<'a, F>,
    scope: &Scope<'a>,
    args: BacktrackArgs<'a, Q>,
) where
    Q: VertexBitSet + 'a,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    let tid = rayon::current_thread_index().unwrap();
    let thread = &mut e.threads[tid].lock().unwrap();

    trace!(
        "start_backtracking: thread_id = {}, m = {}, cand_stack = {}",
        tid,
        args.m,
        CandStackDisplay(&args.cand_stack, &e.cs),
    );

    let query_total_deg = query_vertices(&e.cs)
        .map(|u| query_degree(&e.cs, u))
        .sum::<usize>();

    // Increase ages of given assignments
    for u in 0..args.m.len() {
        thread.assignment_ages[u] += 1;
    }

    let u_bottom = args.m.len() as Vertex;

    let mut s = BacktrackState {
        env: e,
        u_bottom,
        thread,
        scope,
        m: args.m,
        cand_stack: args.cand_stack,
        tree_cands: args.tree_cands,
        local_cand_indices: vec![0; query_vertex_count(&e.cs)],
        skipedge_masks: args.skipedge_masks,
        boundings: args.boundings,
        refinement_backups: Vec::with_capacity(query_total_deg),
        match_count_cache: 0,
        match_increment: 0,
        vertex_nogoods: allocate_vertex_nogoods(&e.cs),
        edge_nogoods: allocate_edge_nogoods(&e.cs, e.core_size),
    };

    let start = Instant::now();

    let _ = search_recursion(&mut s); // Recursion-based search is faster
    //search_loop(&mut s);

    s.thread.running_secs += start.elapsed().as_secs_f64();

    // Sum up match counts that are not yet added
    e.match_count.fetch_add(s.match_increment, SeqCst);
    s.match_increment = 0;

    // Note: This function is called after incrementing `running_count` but
    // decrements it by oneself
    e.running_count.fetch_sub(1, SeqCst);
}
