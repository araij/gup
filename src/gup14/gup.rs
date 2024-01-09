use super::bitset::*;
use super::embedding::ReorderedEmbedding;
use super::filtering::max_nbr_deg;
use super::*;
use std::ops::Range;

use super::backtrack::*;
use super::plan::*;
use crate::graph::*;
use crate::utils::*;
use itertools::Itertools;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::sync::atomic::Ordering::SeqCst;

////////////////////////////////////////////////////////////////////////////////
//
// Gup
//
////////////////////////////////////////////////////////////////////////////////

/// Set of query-agnostic data for a given data graph and options.
pub struct Gup<'a> {
    pub g: &'a Graph,
    pub elist: Edgelist,
    pub label_degs: Vec<Vec<(Label, usize)>>,
    pub of_label: Vec<Vec<Vertex>>,
    pub max_nbr_degs: Vec<usize>,
    pub opt: SearchOptions,
    pub pool: ThreadPool,
}

impl<'a> Gup<'a> {
    pub fn new(g: &'a Graph, opt: SearchOptions) -> Self {
        let pool = ThreadPoolBuilder::new()
            .num_threads(opt.parallelism)
            .build()
            .unwrap();

        Gup {
            g,
            elist: make_edgelist(g),
            label_degs: count_neighbors(g),
            of_label: group_labels(g),
            max_nbr_degs: count_max_neighbor_degrees(g),
            opt: opt,
            pool,
        }
    }

    pub fn search<F>(&self, q: &Graph, report: F) -> (usize, bool)
    where
        F: Fn(&ReorderedEmbedding) -> bool + Sync,
    {
        match q.num_vertices() {
            n if n <= 64 => search::<VertexBitSet64, _>(self, q, report),
            n if n <= 128 => search::<VertexBitSet128, _>(self, q, report),
            n if n <= 256 => search::<VertexBitSet256, _>(self, q, report),
            _ => {
                panic!(
                    "This implementation does not support query graphs with \
                    more than 256 vertices. Sorry for the inconvenience."
                )
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Edgelist
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Edgelist {
    pub n_label: usize,
    pub srcs: Vec<Vertex>,
    pub trgs: Vec<Vertex>,
    sstarts_: Vec<usize>, // (label * label) -> index in `srcs`
    tstarts_: Vec<usize>, // index in `srcs` -> index in `trgs`
}

#[allow(dead_code)]
impl Edgelist {
    pub fn new(
        n_label: usize,
        srcs: Vec<Vertex>,
        trgs: Vec<Vertex>,
        sstarts_: Vec<usize>,
        tstarts_: Vec<usize>,
    ) -> Edgelist {
        assert!(srcs.len() <= trgs.len());
        assert_eq!(sstarts_.len(), n_label * n_label + 1);
        assert_eq!(tstarts_.len(), srcs.len() + 1);
        debug_assert!(sstarts_.iter().is_ordered());
        debug_assert!(tstarts_.iter().is_ordered());
        debug_assert!(sstarts_.iter().all(|&i| i <= srcs.len()));
        debug_assert!(tstarts_.iter().all(|&i| i <= trgs.len()));

        Edgelist {
            n_label,
            srcs,
            trgs,
            sstarts_,
            tstarts_,
        }
    }

    pub fn sindex(&self, ls: Label, lt: Label) -> Range<usize> {
        assert!((ls as usize) < self.n_label);
        assert!((lt as usize) < self.n_label);
        let i = (ls as usize) * self.n_label + (lt as usize);
        self.sstarts_[i]..self.sstarts_[i + 1]
    }

    pub fn srange<'a>(&'a self, ls: Label, lt: Label) -> &'a [Vertex] {
        &self.srcs[self.sindex(ls, lt)]
    }

    pub fn tindex(&self, sindex: usize) -> Range<usize> {
        assert!(sindex < self.srcs.len());
        self.tstarts_[sindex]..self.tstarts_[sindex + 1]
    }

    pub fn trange<'a>(&'a self, sindex: usize) -> &'a [Vertex] {
        &self.trgs[self.tindex(sindex)]
    }

    pub fn len(&self) -> usize {
        self.trgs.len()
    }

    // Renamed from overload: `size_t size(vlabel_id src, vlabel_id trg)`
    pub fn edge_count(&self, src: Label, trg: Label) -> usize {
        let r = self.sindex(src, trg);
        self.tstarts_[r.end] - self.tstarts_[r.start]
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Public Free Functions
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Private Free Functions
//
////////////////////////////////////////////////////////////////////////////////

fn search<Q, F>(data: &Gup, q: &Graph, report: F) -> (usize, bool)
where
    Q: VertexBitSet,
    F: Fn(&ReorderedEmbedding) -> bool + Sync,
{
    let Some(e) = plan::<Q, _>(data, q, report) else {
        return (0, false); // No embedding
    };

    backtrack_parallel::<Q, _>(&e);

    if e.opt.probe {
        show_probe::<Q, _>(&e);
    }

    // `match_count` may exceed the limit due to delay of synchronization
    let n = e.match_count.load(SeqCst).min(e.opt.match_limit);
    (n, e.timer.is_over())
}

fn verify_edgelist(g: &Graph, el: &Edgelist) {
    let mut eledges = Vec::with_capacity(el.trgs.len());

    for ls in 0..(g.num_labels() as Label) {
        for lt in 0..(g.num_labels() as Label) {
            for (is, &s) in el.sindex(ls, lt).zip(el.srange(ls, lt)) {
                assert_eq!(g.vertex_label(s), ls);
                for &t in el.trange(is) {
                    assert_eq!(g.vertex_label(t), lt);
                    eledges.push((s, t));
                }
            }
        }
    }

    let mut gedges = Vec::with_capacity(el.trgs.len());
    for s in g.vertices() {
        for t in g.out_neighbors(s) {
            gedges.push((s, t));
        }
    }

    assert_eq!(eledges.len(), gedges.len());

    eledges.sort();
    gedges.sort();
    assert_eq!(&eledges, &gedges);
}

fn make_edgelist(g: &Graph) -> Edgelist {
    //
    // Convert graph to an edge list
    //
    let mut edges = Vec::with_capacity(g.num_edges());
    for v in g.vertices() {
        for v_ in g.out_neighbors(v) {
            edges.push((v, v_));
        }
    }
    assert!(edges.len() == g.num_edges());

    //
    // Sort edges by
    // (source label, target label, source vertex ID, target vertex ID).
    //
    edges.sort_by(|e0, e1| {
        let ls0 = g.vertex_label(e0.0);
        let ls1 = g.vertex_label(e1.0);
        if ls0 != ls1 {
            return ls0.cmp(&ls1);
        }

        let lt0 = g.vertex_label(e0.1);
        let lt1 = g.vertex_label(e1.1);
        if lt0 != lt1 {
            return lt0.cmp(&lt1);
        }

        e0.cmp(&e1)
    });

    //
    // Split sources and targets of edges
    // and group-by source vertex IDs
    // e.g. [(0, 3), (0, 4), (0, 7), (1, 4), (1, 6), (2, 4), (2, 5), (2, 6)]
    //      ==> [(0, (3, 4, 7)), (1, (4, 6)), (2, (4, 5, 6))]
    // where vertex 0-2 and vertex 3-7 have the same label, respectively
    //
    let mut srcs = Vec::with_capacity(edges.len());
    let mut trgs = Vec::with_capacity(edges.len());
    let mut tstarts = Vec::with_capacity(edges.len() + 1);
    tstarts.push(0);

    let mut i = 0;
    while i < edges.len() {
        let mut j = i;
        while j < edges.len()
            && edges[j].0 == edges[i].0
            && g.vertex_label(edges[j].1) == g.vertex_label(edges[i].1)
        {
            j += 1;
        }
        assert!(j > i);

        srcs.push(edges[i].0);
        trgs.extend(edges[i..j].iter().map(|&e| e.1));
        tstarts.push(j);
        i = j;
    }

    srcs.shrink_to_fit();
    trgs.shrink_to_fit();
    tstarts.shrink_to_fit();

    assert!(srcs.len() <= edges.len());
    assert!(trgs.len() <= edges.len());
    assert!(srcs.len() <= trgs.len());
    assert!(tstarts.len() == srcs.len() + 1);

    //
    // Make an index from (source label, target label) to the index in `srcs`
    //
    let mut sstarts = Vec::with_capacity(g.num_labels() * g.num_labels() + 1);
    sstarts.push(0);

    for ls in 0..(g.num_labels() as Label) {
        for lt in 0..(g.num_labels() as Label) {
            let mut i = *sstarts.last().unwrap();
            while i < srcs.len()
                && g.vertex_label(srcs[i]) == ls
                && g.vertex_label(trgs[tstarts[i]]) == lt
            {
                i += 1;
            }
            sstarts.push(i);
        }
    }
    assert_eq!(sstarts.len(), g.num_labels() * g.num_labels() + 1);

    let _g = Edgelist::new(g.num_labels(), srcs, trgs, sstarts, tstarts);
    if cfg!(debug_assertions) {
        verify_edgelist(g, &_g);
    }

    _g
}

fn count_neighbors(g: &Graph) -> Vec<Vec<(Label, usize)>> {
    g.vertices()
        .map(|v| {
            let mut nbs: Vec<_> = g.out_neighbors(v).collect();
            nbs.sort_by_key(|&x| g.vertex_label(x));
            nbs.iter()
                .group_by(|&&x| g.vertex_label(x))
                .into_iter()
                .map(|(k, gr)| (k, gr.count()))
                .collect()
        })
        .collect()
}

/// Makes an index: Label -> [Vertex]
fn group_labels(g: &Graph) -> Vec<Vec<Vertex>> {
    let mut ix = vec![vec![]; g.num_labels()];
    let mut nverts = vec![0; g.num_labels()];
    for v in g.vertices() {
        nverts[g.vertex_label(v) as usize] += 1;
    }
    for i in 0..g.num_labels() {
        ix[i].reserve(nverts[i]);
    }
    for v in g.vertices() {
        ix[g.vertex_label(v) as usize].push(v);
    }
    ix
}

/// Returns a mapping: v -> largest degree among v's neighbors
fn count_max_neighbor_degrees(g: &Graph) -> Vec<usize> {
    let mut mnds = Vec::with_capacity(g.num_vertices());
    for v in g.vertices() {
        mnds.push(max_nbr_deg(v, g));
    }
    mnds
}
