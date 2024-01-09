use std::fmt::Formatter;
use std::fmt::{self, Display};
use std::ops::Index;

use itertools::Itertools;

use crate::graph::*;

////////////////////////////////////////////////////////////////////////////////
//
// Embedding
//
////////////////////////////////////////////////////////////////////////////////

const UNUSED_KEY: Vertex = Vertex::MAX;

pub struct Embedding {
    /// u -> data vertex.
    pub vertices: Vec<Vertex>,
    /// Simple hash table from a data vertex to a query vertex.
    pub used: Vec<(Vertex, Vertex)>,
}

impl Embedding {
    pub fn new(query_vertex_count: usize) -> Embedding {
        Embedding {
            vertices: Vec::with_capacity(query_vertex_count),
            used: vec![(UNUSED_KEY, 0); query_vertex_count * 8 + 1],
        }
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn vertex(&self, u: Vertex) -> Vertex {
        self.vertices[u as usize]
    }

    // Necessary for implementing `Index` for `ReorderedEmbedding`
    pub fn vertex_ref(&self, u: Vertex) -> &Vertex {
        &self.vertices[u as usize]
    }

    /// Returns a query vertex already assigned to data vertex `v` (if exists)
    pub fn invert(&self, v: Vertex) -> Option<Vertex> {
        match self.used[(v as usize) % self.used.len()] {
            (UNUSED_KEY, _) => None,
            (v_, u) if v_ == v => Some(u),
            _ => (0..self.len() as Vertex).find(|&u| self.vertex(u) == v),
        }
    }

    pub fn push(&mut self, v: Vertex, _iv: usize) {
        let u = self.len() as Vertex;
        let i = (v as usize) % self.used.len();
        if self.used[i].0 == UNUSED_KEY {
            self.used[i] = (v, u);
        }

        debug_assert!(self.vertices.len() + 1 <= self.vertices.capacity());
        self.vertices.push(v);
    }

    pub fn pop(&mut self) {
        let v = self.vertices.pop().unwrap();
        let i = (v as usize) % self.used.len();
        if self.used[i].0 == v {
            self.used[i].0 = UNUSED_KEY;
        }
    }

    pub fn copy(&self, len: usize) -> Embedding {
        let mut vertices = Vec::with_capacity(self.vertices.capacity());
        vertices.extend(self.vertices[..len].into_iter());

        let used = self
            .used
            .iter()
            .map(|&(v, u)| {
                if u < len as Vertex {
                    (v, u)
                } else {
                    (UNUSED_KEY, 0)
                }
            })
            .collect();

        Embedding { vertices, used }
    }
}

impl Display for Embedding {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_list().entries(self.vertices.iter()).finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// ReorderedEmbedding
//
////////////////////////////////////////////////////////////////////////////////

pub struct ReorderedEmbedding<'a> {
    m: &'a Embedding,
    perm: &'a Vec<Vertex>,
}

impl<'a> ReorderedEmbedding<'a> {
    pub fn new(m: &'a Embedding, perm: &'a Vec<Vertex>) -> Self {
        ReorderedEmbedding { m, perm }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vertex> {
        (0..self.m.len()).map(|u| &self[u as Vertex])
    }

    pub fn to_vec(&self) -> Vec<Vertex> {
        self.iter().cloned().collect_vec()
    }
}

impl<'a> Index<Vertex> for ReorderedEmbedding<'a> {
    type Output = Vertex;

    fn index(&self, u_orig: Vertex) -> &Self::Output {
        let u_reord = *self
            .perm
            .get(u_orig as usize)
            .expect("Vertex ID out of range");
        self.m.vertex_ref(u_reord)
    }
}
