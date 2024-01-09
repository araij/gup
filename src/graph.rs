use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cmp::PartialEq;
use std::iter::once;
use std::ops::Range;

pub const INVALID_VERTEX_ID: Vertex = Vertex::MAX;

pub type VInt = u32;
pub type Vertex = VInt;
pub type Label = VInt;
pub type Edge = (Vertex, Vertex);

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    neighbors: Vec<Vec<Vertex>>,
    labels: Vec<Label>,
    n_edges: usize,
    n_labels: usize,
}

impl Eq for Graph {}

impl<'a> Graph {
    /// Make a graph from a edge list.
    ///
    /// Vertex IDs must be contiguous in `0..vertices.count()`.
    /// Loop edges are removed.
    pub fn from_edges<V, E>(vertices: V, edges: E) -> Graph
    where
        V: IntoIterator,
        V::Item: Borrow<Label>,
        E: IntoIterator,
        E::Item: Borrow<(Vertex, Vertex)>,
    {
        let labels: Vec<Label> =
            vertices.into_iter().map(|x| *x.borrow()).collect();
        let n_labels = 1 + *labels.iter().max().unwrap() as usize;

        let mut edges: Vec<Edge> = edges
            .into_iter()
            .map(|x| *x.borrow())
            .filter(|(s, t)| s != t) // Remove loops
            .flat_map(|(s, t)| once((s, t)).chain(once((t, s))))
            .collect();
        edges.sort();
        edges.dedup();

        let mut neighbors = vec![vec![]; labels.len()];
        for (&v, nb) in edges.iter().group_by(|&(s, _)| s).into_iter() {
            if v as usize >= neighbors.len() {
                panic!("No data for vertex {}", v);
            }
            neighbors[v as usize] = nb.map(|&(_, t)| t).collect();
        }

        Graph {
            neighbors,
            labels,
            n_edges: edges.len(),
            n_labels,
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.neighbors.len()
    }

    /// Returns the number of edges
    ///
    /// Note that edges (u, v) and (v, u) are distinguished in counting.
    #[allow(dead_code)]
    pub fn num_edges(&self) -> usize {
        self.n_edges
    }

    pub fn num_labels(&self) -> usize {
        self.n_labels
    }

    pub fn vertices(&self) -> Range<Vertex> {
        0..(self.neighbors.len() as Vertex)
    }

    pub fn vertex_label(&self, v: Vertex) -> Label {
        self.labels[v as usize]
    }

    pub fn out_degree(&self, v: Vertex) -> VInt {
        self.neighbors[v as usize].len() as VInt
    }

    pub fn out_neighbors<'b>(
        &'b self,
        v: Vertex,
    ) -> impl Iterator<Item = Vertex> + 'b {
        self.neighbors[v as usize].iter().cloned()
    }
}

pub fn is_isomorphic_embedding(
    q: &Graph,
    g: &Graph,
    m: &[Vertex],
) -> Result<(), String> {
    if m.len() != q.num_vertices() {
        return Err(format!(
            "Different size: query = {}, embedding = {}",
            q.num_vertices(),
            m.len(),
        ));
    }

    for (v0, v1) in m.iter().sorted().tuple_windows() {
        if v0 == v1 {
            let u0 = m.iter().position(|v| v == v0).unwrap();
            let u1 = m.iter().position(|v| v == v0).unwrap();
            return Err(format!(
                "Both u{} and u{} are mapped into v{}",
                u0, u1, v0
            ));
        }
    }

    for u in q.vertices() {
        let v = m[u as usize];
        if q.vertex_label(u) != g.vertex_label(v) {
            return Err(format!(
                "u{} (label {}) is mapped to v{} (label {})",
                u,
                q.vertex_label(u),
                v,
                g.vertex_label(v)
            ));
        }
    }

    // All edges
    for u in q.vertices() {
        for u_ in q.out_neighbors(u) {
            if u <= u_ {
                let v = m[u as usize];
                let v_ = m[u_ as usize];
                if g.neighbors[v as usize].binary_search(&v_).is_err() {
                    return Err(format!(
                        "Query edge ({u}, {u_}) is mapped to nonexistent data \
                        edge ({v}, {v_})"
                    ));
                }
            }
        }
    }

    Ok(())
}

//------------------------------------------------------------------------------
//
// Breadth-first search
//
//------------------------------------------------------------------------------

#[allow(dead_code)]
pub struct BfsMetrics {
    pub lvs: Vec<VInt>,               // v -> level from the root
    pub parents: Vec<Option<Vertex>>, // v -> Some(parent), or None if v = root
    pub ord: Vec<Vertex>,             // Visiting order
    pub nths: Vec<usize>,             // v -> #vertices visited before v
}

pub fn breadth_first(g: &Graph, root: Vertex) -> BfsMetrics {
    let mut parents = vec![None; g.num_vertices()];
    let mut lvs = vec![0; g.num_vertices()];
    let mut ord = Vec::with_capacity(g.num_vertices());
    let mut lv = 0;

    lvs[root as usize] = lv;
    ord.push(root);
    let mut frontier = 0..1;

    while frontier.len() > 0 {
        lv += 1;
        for i in frontier.clone() {
            let v = ord[i];
            for v_ in g.out_neighbors(v) {
                if v_ != root && parents[v_ as usize].is_none() {
                    parents[v_ as usize] = Some(v);
                    lvs[v_ as usize] = lv;
                    ord.push(v_);
                }
            }
        }
        frontier = frontier.end..ord.len();
    }

    assert_eq!(ord.len(), g.num_vertices());
    let mut nths = vec![0; g.num_vertices()];
    for i in 0..g.num_vertices() {
        nths[ord[i] as usize] = i;
    }

    BfsMetrics {
        lvs,
        parents,
        ord,
        nths,
    }
}

//------------------------------------------------------------------------------
//
// k-core Decomposition
//
//------------------------------------------------------------------------------

//
// k-core decomposition [Batagelj and Zaversnik, CoRR'03]
//
pub fn bz_kcore(g: &Graph) -> Vec<usize> {
    let n = g.num_vertices();
    let mut deg = vec![0; g.num_vertices()];
    let mut md = 0; // max degree
    for v in 0..n {
        deg[v] = g.out_degree(v as Vertex) as usize;
        md = std::cmp::max(md, deg[v]);
    }

    let mut bin = vec![0; md + 1];
    for v in 0..n {
        bin[deg[v]] += 1;
    }
    // partial_sum
    for i in 1..bin.len() {
        bin[i] += bin[i - 1];
    }

    let mut vert = vec![0; n];
    let mut pos = vec![0; n];
    let mut ihead = bin.clone();
    for v in 0..n {
        pos[v] = ihead[deg[v] - 1];
        ihead[deg[v] - 1] += 1;
        vert[pos[v]] = v as Vertex;
    }

    for i in 0..n {
        let v = vert[i];
        for u in g.out_neighbors(v) {
            let u = u as usize;
            let v = v as usize;
            if deg[u as usize] > deg[v] {
                let w = vert[bin[deg[u] - 1]] as usize;
                if u != w {
                    vert.swap(pos[u], pos[w]);
                    pos.swap(u, w);
                }
                bin[deg[u] - 1] += 1;
                deg[u] -= 1;
            }
        }
    }

    deg
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_graph_from_simple_edges() {
        //    v0 (0)
        //      /   \
        // v1 (1)---(2) v2
        //
        // Contiguous vertex IDs and label IDs
        let vertices = [0, 1, 2];
        // Sorted edges without duplicates
        let edges = [(0, 1), (0, 2), (1, 2)];

        let g = Graph::from_edges(&vertices, &edges);
        assert_eq!(g.neighbors, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
        assert_eq!(g.labels, vec![0, 1, 2]);
        assert_eq!(g.n_edges, 6); // x2 for bidirectional edges
    }

    #[test]
    fn test_graph_from_complicated_edges() {
        //    v0 (3)
        //      /   \
        // v1 (9)---(2) v2
        //
        // Contiguous vertex IDs and non-contiguous label IDs
        let vertices = [3, 9, 2];
        // Unsorted edges with duplicates and loops
        let edges = [(2, 0), (1, 2), (0, 2), (0, 1), (1, 2), (2, 2)];

        let g = Graph::from_edges(&vertices, &edges);
        assert_eq!(g.neighbors, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
        assert_eq!(g.labels, vec![3, 9, 2]);
        assert_eq!(g.n_edges, 6); // x2 for bidirectional edges
    }

    #[test]
    fn test_breadth_first() {
        //    v0 (0)
        //      /   \
        // v1 (0)---(0) v2
        //           |
        //       v3 (0)
        let vertices = [0, 0, 0, 0];
        let edges = [(0, 1), (0, 2), (1, 2), (2, 3)];
        let g = Graph::from_edges(vertices, edges);

        let b = breadth_first(&g, 1);

        assert_eq!(b.lvs[0], 1);
        assert_eq!(b.lvs[1], 0);
        assert_eq!(b.lvs[2], 1);
        assert_eq!(b.lvs[3], 2);

        assert_eq!(b.parents[0], Some(1));
        assert_eq!(b.parents[1], None);
        assert_eq!(b.parents[2], Some(1));
        assert_eq!(b.parents[3], Some(2));

        assert!(
            itertools::equal(&b.ord, &[1, 0, 2, 3])
                || itertools::equal(&b.ord, &[1, 2, 0, 3])
        );

        assert!(b.nths[0] == 1 || b.nths[0] == 2);
        assert_eq!(b.nths[1], 0);
        assert!(b.nths[2] == 1 || b.nths[2] == 2);
        assert_eq!(b.nths[3], 3);
    }

    #[test]
    fn test_bz_kcore() {
        //    v0 (0)
        //      /   \
        // v1 (0)---(0) v2
        //     |  X  |
        // v3 (0)---(0) v4
        //           |
        //          (0) v5
        let vertices = [0, 0, 0, 0, 0, 0];
        let edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
        ];
        let g = Graph::from_edges(vertices, edges);
        let core = bz_kcore(&g);

        assert_eq!(core[0], 2);
        assert_eq!(core[1], 3);
        assert_eq!(core[2], 3);
        assert_eq!(core[3], 3);
        assert_eq!(core[4], 3);
        assert_eq!(core[5], 1);
    }
}
