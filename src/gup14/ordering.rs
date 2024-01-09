use super::cand_space::*;
use crate::graph::*;
use crate::utils::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

//------------------------------------------------------------------------------
//
// Ordering optimization
//
//------------------------------------------------------------------------------

fn order_core(
    q: &Graph,
    qtree: &BfsMetrics,
    core: &Vec<usize>,
    cs: &CandSpace,
    ord: &mut Vec<Vertex>,
    ordered: &mut Vec<bool>,
) {
    assert_eq!(ord.len(), 1); // `ord` contains only the root
    assert!(ordered[*ord.first().unwrap() as usize]); // The root is ordered
    assert!(ord.capacity() >= q.num_vertices()); // Avoid realloc

    //
    // u -> u' -> (connection strength between u and the candidates of u')
    //
    let mut weights = vec![vec![0.0; q.num_vertices()]; q.num_vertices()];
    for u in q.vertices() {
        if core[u as usize] >= 2 {
            // Calculate only for vertices in the 2-core
            for (_u, nbrs) in &cs[u as usize].neighbors {
                let _u = *_u;
                // Traverse from the top to the bottom; otherwise paths in the
                // CS might be broken.
                if core[_u as usize] >= 2
                    && qtree.ord[u as usize] < qtree.ord[_u as usize]
                {
                    // The comparison on `qtree.ord` prevents from duplicate
                    // computation of the same value
                    let w = (0..cs[u as usize].candidates.len())
                        .map(|iv| nbrs[iv].len())
                        .sum::<usize>() as f64
                        / cs[u as usize].candidates.len() as f64
                        / cs[_u as usize].candidates.len() as f64;
                    weights[u as usize][_u as usize] = w;
                    weights[_u as usize][u as usize] = w;
                }
            }
        }
    }

    let mut pq = BinaryHeap::new(); // Min-heap

    let mut queued = vec![false; q.num_vertices()];
    let mut selectivity = vec![1.0; q.num_vertices()];

    let cost =
        |u, ordered: &Vec<bool>, queued: &Vec<_>, selectivity: &Vec<_>| {
            let mut c = selectivity[u as usize]
                * cs[u as usize].candidates.len() as f64;
            for _u in q.out_neighbors(u) {
                if queued[_u as usize] && !ordered[_u as usize] {
                    c *= weights[u as usize][_u as usize]
                        * selectivity[_u as usize];
                }
            }
            c
        };

    let mut u = *ord.first().unwrap(); // root
    loop {
        for _u in q.out_neighbors(u) {
            if core[_u as usize] >= 2 && !ordered[_u as usize] {
                queued[_u as usize] = true;
                selectivity[_u as usize] *= weights[u as usize][_u as usize];
            }
        }
        // The loop of the cost calculation need to be separated because it
        // depends on `queued` and `selectivity`.
        for _u in q.out_neighbors(u) {
            if core[_u as usize] >= 2 && !ordered[_u as usize] {
                pq.push(Reverse((
                    Total(cost(_u, ordered, &queued, &selectivity)),
                    _u,
                )));
            }
        }

        // A vertex might be inserted multiple times with different priorities
        loop {
            if pq.len() == 0 {
                return;
            }
            u = pq.pop().unwrap().0 .1;
            if !ordered[u as usize] {
                break;
            }
        }

        ord.push(u);
        ordered[u as usize] = true;
    }
}

fn order_trees(
    q: &Graph,
    qtree: &BfsMetrics,
    cs: &CandSpace,
    ord: &mut Vec<Vertex>,
    ordered: &mut Vec<bool>,
) {
    //
    // Prioritize each query vertex based on the estimated number of embeddings
    // under the assumption that the query graph is a spanning tree and
    // non-injective embeddings are allowed.
    //
    let mut nembeds = vec![0.0; q.num_vertices()];

    fn countembed(
        u: Vertex,
        qtree: &BfsMetrics,
        cs: &CandSpace,
        ordered: &mut Vec<bool>,
        nembeds: &mut Vec<f64>,
    ) -> Vec<usize> {
        let mut npaths = vec![1; cs[u as usize].candidates.len()];

        for &(u_, ref nbrs) in &cs[u as usize].neighbors {
            if qtree.parents[u_ as usize].map_or(true, |x| x != u)
                || ordered[u_ as usize]
            {
                continue;
            }

            let mut _npaths = countembed(u_, qtree, cs, ordered, nembeds);
            let mut npsum = 0;
            for iv in 0..cs[u as usize].candidates.len() {
                let np =
                    nbrs[iv].iter().map(|&_iv| _npaths[_iv]).sum::<usize>();
                npaths[iv] *= np;
                npsum += np;
            }

            assert!(0.0 <= nembeds[u_ as usize] && nembeds[u_ as usize] <= 0.0);
            nembeds[u_ as usize] =
                npsum as f64 / cs[u as usize].candidates.len() as f64;
        }

        npaths
    }

    let mut pq = BinaryHeap::new(); // Min-heap

    for u in q.vertices() {
        if ordered[u as usize]
            && q.out_neighbors(u).any(|u| !ordered[u as usize])
        {
            countembed(u, qtree, cs, ordered, &mut nembeds);
            for _u in q.out_neighbors(u) {
                if !ordered[_u as usize] {
                    pq.push(Reverse((0, Total(nembeds[_u as usize]), _u)));
                }
            }
        }
    }

    while pq.len() > 0 {
        let Reverse((d, _, u)) = pq.pop().unwrap();

        assert!(!ordered[u as usize]);
        ord.push(u);
        ordered[u as usize] = true;
        for _u in q.out_neighbors(u) {
            if !ordered[_u as usize] {
                pq.push(Reverse((d - 1, Total(nembeds[_u as usize]), _u)));
            }
        }
    }
}

pub fn optimize_order(
    q: &Graph,
    cs: &CandSpace,
    root: Vertex,
    core: &Vec<usize>,
) -> Vec<Vertex> {
    // `root` must be in the 2-core (if the 2-core exists)
    debug_assert!(*core.iter().max().unwrap() <= 1 || core[root as usize] >= 2);

    let mut ordered = vec![false; q.num_vertices()];
    let mut ord = Vec::with_capacity(q.num_vertices());

    // The first vertex is always `root`
    ord.push(root);
    ordered[root as usize] = true;

    let qtree = breadth_first(q, root);
    order_core(q, &qtree, core, cs, &mut ord, &mut ordered);
    order_trees(q, &qtree, cs, &mut ord, &mut ordered);

    //dump_as_dot(q, root, &std::cerr);

    assert_eq!(ord.len(), q.num_vertices());
    debug_assert!(q.vertices().all(|u| ord.contains(&u)));

    ord
}

/// Make a permutation from an ordering.
///
/// Input: `order[reordered ID] = original ID`.  
/// Output: `perm[original ID] = reordered ID`.
pub fn to_permutation(order: &Vec<Vertex>) -> Vec<Vertex> {
    let mut perm = vec![0; order.len()];
    for i in 0..order.len() {
        perm[order[i] as usize] = i as Vertex;
    }
    perm
}

#[allow(dead_code)]
pub fn reorder<T: Clone + Copy>(xs: Vec<T>, ord: &[Vertex]) -> Vec<T> {
    ord.iter().map(|&i| xs[i as usize]).collect()
}
