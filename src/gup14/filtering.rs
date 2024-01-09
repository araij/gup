use std::iter::repeat_with;

use super::bitset::is_superset;
use super::bitset::VertexBitSet;
use super::cand_space::*;
use super::gup::*;
use crate::graph::*;
use crate::utils::*;
use itertools::Itertools;

const PROPAGATION_COUNT: u32 = 1;

////////////////////////////////////////////////////////////////////////////////
//
// Public Free Functions
//
////////////////////////////////////////////////////////////////////////////////

pub fn prune<'a, Q: VertexBitSet>(
    q: &Graph,
    core: &Vec<usize>,
    g: &Graph,
    gix: &'a Gup,
) -> (Vertex, Vec<Q>, Vec<&'a [Vertex]>) {
    let ge = &gix.elist;
    let nvert = gix.label_degs.len();
    let mut qnbrs = group_neighbors(q);

    // u -> [initial candidate vertices of `u`].
    // Assuming that query vertex `u` has label L and the neighbors of `u` have
    // labels M_0, M_1, ..., M_k, the candidate vertices of `u` must appear as a
    // source vertex of an edge that connects label L and label M_0, M_1, ...,
    // or M_k. Thus, here we find M_i that yields the fewest number of the edges
    // and use their source vertices as an initial candidate vertex set.
    let hints: Vec<_> = q
        .vertices()
        .map(|u| {
            let l = q.vertex_label(u);
            let (_l, _) = qnbrs[u as usize]
                .iter()
                .min_by_key(|x| ge.sindex(l, x.0).len())
                .unwrap();
            ge.srange(l, *_l)
        })
        .collect();

    // v -> [query vertices that `v` can match]
    let mut matchas = vec![Q::ZERO; nvert];
    // u -> #candidate vertices (in consistency with `matchas`)
    let mut nhintcand = vec![0; q.num_vertices()];
    // Apply filters to obtain `matchas` and `nhintcand`
    for u in q.vertices() {
        let mnd = max_nbr_deg(u, q);
        for &v in hints[u as usize] {
            if g.out_degree(v) >= q.out_degree(u)
                && gix.max_nbr_degs[v as usize] >= mnd
                && has_neighbors(
                    &qnbrs[u as usize],
                    &gix.label_degs[v as usize],
                )
            {
                matchas[v as usize].set(u as usize, true);
                nhintcand[u as usize] += 1;
            }
        }
    }

    let uroot = select_root(q, core, &nhintcand);
    let qtree = breadth_first(q, uroot);

    //
    // Top-down and bottom-up refinement
    //
    // Sort neighbors to begin propagation from an adjacent label that has fewer
    // adjacent vertices because it is more difficult to meet the match
    // condition for such a label (i.e., easy to prune).
    // Note that we cannot sort `qnbrs` before here because `has_neighbors`
    // requires `qnbrs` to be sorted by a label ID.
    for u in q.vertices() {
        let l = q.vertex_label(u);
        qnbrs[u as usize].sort_by_key(|x| ge.edge_count(l, x.0));
    }

    for _ in 0..PROPAGATION_COUNT {
        for &u in &qtree.ord {
            propagate::<1, Q>(
                q,
                &qtree,
                ge,
                u,
                &qnbrs[u as usize],
                &mut matchas,
            );
        }
        for &u in qtree.ord.iter().rev() {
            propagate::<2, Q>(
                q,
                &qtree,
                ge,
                u,
                &qnbrs[u as usize],
                &mut matchas,
            );
        }
    }

    (uroot, matchas, hints)
}

// Note that the constructed candidate space may contain candidate vertices that
// is not connected to others because the constant number of iterations of
// top-down/bottom-up refinement may fail to prune all of the such vertices.
pub fn construct_cs<Q: VertexBitSet>(
    q: &Graph,
    g: &Graph,
    ge: &Edgelist,
    matchas: &Vec<Q>,
    hints: &Vec<&[Vertex]>,
) -> Vec<Node> {
    let mut csnodes: Vec<_> = repeat_with(|| Node {
        candidates: vec![],
        neighbors: vec![],
    })
    .take(q.num_vertices())
    .collect();
    for u in q.vertices() {
        csnodes[u as usize]
            .candidates
            .reserve(hints[u as usize].len());
        csnodes[u as usize]
            .neighbors
            .reserve(q.out_degree(u) as usize);
    }

    for v in g.vertices() {
        for u in matchas[v as usize].iter_ones() {
            csnodes[u].candidates.push(v);
        }
    }

    for u in q.vertices() {
        assert!(
            csnodes[u as usize].candidates.len() <= hints[u as usize].len()
        );
        csnodes[u as usize].candidates.shrink_to_fit();
    }

    let invalidix = usize::MAX;
    let mut csixs = vec![invalidix; g.num_vertices()];
    for u in q.vertices() {
        for (i, &v) in csnodes[u as usize].candidates.iter().enumerate() {
            csixs[v as usize] = i;
        }

        for _u in q.out_neighbors(u) {
            let mut nbrs = SplitVec::with_capacity(
                csnodes[_u as usize].candidates.len(),
                ge.edge_count(q.vertex_label(_u), q.vertex_label(u)),
            );
            for &_v in &csnodes[_u as usize].candidates {
                // TODO: Optimize iteration over neighbors of a specific label,
                // e.g., `g.out_neighbors_of_label(_v, q.vertex_label(u))`
                for v in g.out_neighbors(_v) {
                    if csixs[v as usize] != invalidix {
                        nbrs.push(csixs[v as usize]);
                    }
                }
                nbrs.close();
                debug_assert!(nbrs[nbrs.len() - 1].iter().is_ordered());
            }
            nbrs.shrink_to_fit();
            csnodes[_u as usize].neighbors.push((u, nbrs));
        }

        for &v in &csnodes[u as usize].candidates {
            csixs[v as usize] = invalidix;
        }
    }

    debug_assert!(q.vertices().all(|u| {
        csnodes[u as usize].neighbors.len() == q.out_degree(u) as usize
    }));
    // Symmetricity is required for changing the root in reordering
    debug_assert!(is_symmetric_cs(&csnodes));

    csnodes
}

/// Returns a mapping: v -> g -> largest degree among v's neighbors
pub fn max_nbr_deg(v: Vertex, g: &Graph) -> usize {
    g.out_neighbors(v)
        .map(|x| g.out_degree(x) as usize)
        .max()
        .unwrap_or(0)
}

////////////////////////////////////////////////////////////////////////////////
//
// Private Free Functions
//
////////////////////////////////////////////////////////////////////////////////

//
// Return: u -> [(adjacent label, [adjacent vertices])]
//
fn group_neighbors(q: &Graph) -> Vec<Vec<(Label, Vec<Vertex>)>> {
    let mut nbrs = vec![];
    let mut ret = Vec::with_capacity(q.num_vertices());

    for u in q.vertices() {
        nbrs.clear();
        nbrs.reserve(q.out_degree(u) as usize);
        nbrs.extend(q.out_neighbors(u));

        let mut grps = Vec::with_capacity(nbrs.len());

        nbrs.sort_by_key(|&u_| q.vertex_label(u_));
        for (k, us) in &nbrs.iter().cloned().group_by(|&u_| q.vertex_label(u_))
        {
            grps.push((k, us.collect()));
        }
        grps.shrink_to_fit();

        ret.push(grps);
    }

    ret
}

/// Perform neighborhood label frequency filtering (NLF) and returns `true` if
/// data vertex `v` can be a candidate vertex of query vertex `u`,  
fn has_neighbors(
    unbr: &Vec<(Label, Vec<Vertex>)>,
    vnbr: &Vec<(Label, usize)>,
) -> bool {
    debug_assert!(unbr.iter().map(|x| x.0).is_ordered());
    debug_assert!(vnbr.iter().map(|x| x.0).is_ordered());

    let mut uit = unbr.iter().peekable();
    let mut vit = vnbr.iter().peekable();

    while uit.peek().is_some() {
        if vit.peek().is_none() || uit.peek().unwrap().0 < vit.peek().unwrap().0
        {
            return false;
        } else if uit.peek().unwrap().0 == vit.peek().unwrap().0 {
            if uit.peek().unwrap().1.len() > vit.peek().unwrap().1 {
                return false;
            }
            uit.next();
        }
        vit.next();
    }

    true
}

fn select_root_path(q: &Graph, nhintcand: &Vec<VInt>) -> Vertex {
    //
    // If the query is a path, beggining the search from the endpoint of the
    // path will offer better performance because nogoods will live longer.
    //
    // If `q` is a path, it has only two degree-1 endpoints
    debug_assert_eq!(q.vertices().filter(|&u| q.out_degree(u) == 1).count(), 2);

    let mut root = None;
    for u in q.vertices() {
        if q.out_degree(u) == 1 {
            if root.is_none()
                || nhintcand[u as usize] < nhintcand[root.unwrap() as usize]
            {
                root = Some(u);
            }
        }
    }

    root.unwrap()
}

fn select_root_core(
    q: &Graph,
    core: &Vec<usize>,
    nhintcand: &Vec<VInt>,
) -> Vertex {
    let coremax = *core.iter().max().unwrap();

    for k in (1..=coremax).rev() {
        // Select a vertex as a root if it have the largest core number in `q`
        // and the smallest penalty defined as follows.
        let mut ps: Vec<_> = q
            .vertices()
            .filter(|&u| core[u as usize] == coremax)
            .map(|u| {
                let coredeg = q
                    .out_neighbors(u)
                    // Compute a degree in the `k`-core
                    .filter(|&u_| core[u_ as usize] >= k)
                    .count();
                let penalty = nhintcand[u as usize] as f64 / coredeg as f64;
                (u, Total(penalty))
            })
            .collect();

        // Return a vertex of the minimum penalty.
        // If multiple vertices take the minimum value, retry with smaller `k`.
        let minpenalty = ps.iter().map(|x| x.1).min().unwrap();
        ps.retain(|x| x.1 == minpenalty);
        assert!(ps.len() > 0);
        if ps.len() == 1 || k == 1 {
            return ps[0].0;
        }
    }

    unreachable!();
}

fn select_root(q: &Graph, core: &Vec<usize>, nhintcand: &Vec<VInt>) -> Vertex {
    // If the degrees of all the query vertices are less than or equal to two,
    // the query graph must be a path or a circle.
    // If it is a circle, the whole graph becomes 2-core.
    // Therefore, if even single vertex is not in 2-core, the query graph is
    // determined to be a path.
    if core[0] == 1 && q.vertices().all(|u| q.out_degree(u) <= 2) {
        select_root_path(q, nhintcand)
    } else {
        select_root_core(q, core, nhintcand)
    }
}

//
// Mode = 0 : bi-directional
// Mode = 1 : top-down
// Mode = 2 : bottom-up
//
fn propagate<const MODE: usize, Q: VertexBitSet>(
    q: &Graph,
    qtree: &BfsMetrics,
    ge: &Edgelist,
    u: Vertex,
    qnbrs: &Vec<(Label, Vec<Vertex>)>,
    matchas: &mut Vec<Q>,
) -> bool {
    let mut modified = false;

    for i in 0..qnbrs.len() {
        let nbrlab = qnbrs[i].0;
        let nbrs = &qnbrs[i].1; // Neighbors whose label is `nbrlab`

        // A data vertex becomes a candidate if it is adjacent to other data
        // vertices that match the query vertices included in `required`.
        let mut nnbr = 0;
        let mut required = Q::ZERO;
        for &_u in nbrs {
            if (MODE == 0)
                || (MODE == 1
                    && qtree.nths[_u as usize] < qtree.nths[u as usize])
                || (MODE == 2
                    && qtree.nths[_u as usize] > qtree.nths[u as usize])
            {
                nnbr += 1;
                required.set(_u as usize, true);
            }
        }

        // Omit the check if `nbr` does not exist for top-down or bottom-up
        if nnbr == 0 {
            continue;
        }

        for six in ge.sindex(q.vertex_label(u), nbrlab) {
            let s = ge.srcs[six];
            if !matchas[s as usize][u as usize] {
                continue;
            }

            let mut n = 0;
            let mut m = Q::ZERO;
            for tix in ge.tindex(six) {
                let t = ge.trgs[tix];

                if (matchas[t as usize] & required).any() {
                    n += 1;
                    m |= matchas[t as usize];
                    if n >= nnbr && is_superset(&m, &required) {
                        break; // Escape early if the match condition is met
                    }
                }
            }

            // `s` can match `u` only if `s` is adjacent to data vertices that
            // match the query vertices in `required`.
            if n < nnbr || !is_superset(&m, &required) {
                matchas[s as usize].set(u as usize, false);
                modified = true;
            }
        }
    }

    modified
}

/// Check if CS is symmetric
///
/// Specifically, if candidate `(u, v)` has an edge to candidate `(_u, _v)`,
/// `(_u, _v)` also has an edge to `(u, v)`.
fn is_symmetric_cs(cs: &Vec<Node>) -> bool {
    let qsize = cs.len();
    let mut downedges = vec![];
    let mut upedges = vec![];

    for u in 0..(qsize as Vertex) {
        for (_u, nbrs) in &cs[u as usize].neighbors {
            let _u = *_u;
            for (iv, v) in cs[u as usize].candidates.iter().copied().enumerate()
            {
                for &_iv in &nbrs[iv] {
                    let _v = cs[_u as usize].candidates[_iv];
                    if u < _u {
                        downedges.push((v, _v));
                    } else {
                        upedges.push((_v, v));
                    }
                }
            }
        }
    }

    downedges.sort_unstable();
    upedges.sort_unstable();

    downedges == upedges
}
