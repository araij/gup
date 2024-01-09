use crate::graph::*;
use crate::utils::*;
use log::info;
use std::io::Write;

////////////////////////////////////////////////////////////////////////////////
//
// Node and CandSpace
//
////////////////////////////////////////////////////////////////////////////////

/// Candidate space node.
///
/// Candidate space node contains data of the candidate vertices of a specific
/// query vertex.
#[derive(Default)]
pub struct Node {
    /// Candidate vertices.
    ///
    /// Suppose that this candidate space node is for query vertex `u`.
    /// If candidate vertex `v` is `i`-th element of this vector, we refer to
    /// `i` as a *candidate index* of `v` in the candidate-vertex set of `u`.
    pub candidates: Vec<Vertex>, // iv -> v

    /// Neighbor of each candidate vertex.
    ///
    /// Suppose that this Node is for query vertex `u` and let `(u_, inbsof)` in
    /// `neighbors`. Then
    /// - `u_` is a query vertex adjacent to `u`.
    /// - `inbsof[i]` is a slice of candidate indices of the adjacent candidate
    ///   vertices of `candidates[i]` in the candidate-vertex set of `u_`.
    pub neighbors: Vec<(Vertex, SplitVec<usize>)>, // [(u, (iv -> inbs)@inbsof)]
}

/// Candidate space.
///
/// The `u`-th element is a candidate space node for query vertex `u`.
pub type CandSpace = Vec<Node>;

////////////////////////////////////////////////////////////////////////////////
//
// Public Free Functions
//
////////////////////////////////////////////////////////////////////////////////

pub fn query_vertices(cs: &CandSpace) -> impl Iterator<Item = Vertex> {
    0..cs.len() as Vertex
}

pub fn query_vertex_count(cs: &CandSpace) -> usize {
    cs.len()
}

pub fn query_degree(cs: &CandSpace, u: Vertex) -> usize {
    cs[u as usize].neighbors.len()
}

pub fn query_neighbors<'a>(
    cs: &'a CandSpace,
    u: Vertex,
) -> impl Iterator<Item = Vertex> + 'a {
    cs[u as usize].neighbors.iter().map(|&(u_, _)| u_)
}

pub fn cand_count(cs: &CandSpace, u: Vertex) -> usize {
    cs[u as usize].candidates.len()
}

pub fn cands(cs: &CandSpace, u: Vertex) -> &[Vertex] {
    cs[u as usize].candidates.as_slice()
}

/// BE CAREFUL: this function takes O(d(u)) time
#[allow(dead_code)]
pub fn candidate_neighbor_indices<'a>(
    cs: &'a CandSpace,
    u: Vertex,
    iv: usize,
    u_dest: Vertex,
) -> &'a [usize] {
    &cs[u as usize]
        .neighbors
        .iter()
        .find(|(u_, _)| *u_ == u_dest)
        .unwrap()
        .1[iv]
}

pub fn reorder(
    mut cs: CandSpace,
    ord: &Vec<Vertex>,
    perm: &Vec<Vertex>,
) -> CandSpace {
    let qsize = perm.len();
    (0..qsize)
        .map(|unew| {
            let mut node = Node::default();
            std::mem::swap(&mut cs[ord[unew] as usize], &mut node);
            for (uold, _) in &mut node.neighbors {
                *uold = perm[*uold as usize];
            }
            node.neighbors.sort_by_key(|&(u, _)| u);
            node
        })
        .collect()
}

pub fn print_details(q: &Graph, uroot: Vertex, cs: &CandSpace) {
    let mut f = std::fs::File::create("cs.dot").expect("Cannot create file");
    write!(f, "{}", dump_dot(q, cs, uroot)).unwrap();

    info!("u\t#cands\t[(neighbor, #edges)]");
    for u in q.vertices() {
        let mut ss = String::new();
        for p in &cs[u as usize].neighbors {
            let _u = p.0;
            let nbrs = &p.1;

            let mut m = 0;
            for iv in 0..cs[u as usize].candidates.len() {
                m += nbrs[iv].len();
            }
            ss += &format!("({}, {}), ", _u, m);
        }
        if ss.len() > 2 {
            ss.truncate(ss.len() - 2); // remove a redundant `, `
        }
        info!("{}\t{}\t{}", u, cs[u as usize].candidates.len(), ss);
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Private Free Functions
//
////////////////////////////////////////////////////////////////////////////////

fn dump_dot(q: &Graph, cs: &CandSpace, root: Vertex) -> String {
    let mut ss = String::new();
    let qtree = breadth_first(q, root);

    ss += "graph candidatespace {\n";
    ss += "  graph [splines = line];\n";
    ss += "  node [shape = \"plaintext\"];\n";
    ss += "  edge [dir = \"none\"];\n";

    for u in 0..cs.len() {
        ss += &format!("  node_{} [label=<\n", u);
        ss += "<table border=\"0\" cellspacing=\"0\"><tr>\n";
        ss += &format!("  <td><b>u{}</b></td>\n", u);
        ss += "  <td>\n";
        ss +=
            "    <table border=\"0\" cellborder=\"1\" cellspacing=\"0\"><tr>\n";

        for i in 0..cs[u].candidates.len() {
            ss += &format!(
                "      <td width=\"20px\" port=\"{}\">{}</td>\n",
                i, cs[u].candidates[i],
            );
        }

        ss += "    </tr></table>\n";
        ss += "  </td>\n";
        // Add a hidden column to prevent an edge connected to the right border
        ss += "  <td width=\"0px\"></td>\n";
        ss += "</tr></table>>];\n";
    }

    for u in 0..cs.len() {
        for i in 0..cs[u].candidates.len() {
            for adj in &cs[u].neighbors {
                if qtree.lvs[u] <= qtree.lvs[adj.0 as usize] {
                    for _i in &adj.1[i] {
                        ss += &format!(
                            "  node_{}:{} -- node_{}:{};\n",
                            u, i, adj.0, _i,
                        );
                    }
                }
            }
        }
    }

    ss + "}\n"
}
