use std::fmt;

use super::bitset::*;
use super::cand_space::*;
use crate::graph::*;
use crate::utils::*;

#[derive(Default, Debug, Clone)]
pub struct Nogood<Q: VertexBitSet> {
    u: Vertex,
    age: usize,
    mask: Q,
}

impl<Q: VertexBitSet> Nogood<Q> {
    pub fn permanent() -> Self {
        Nogood {
            u: 0,
            age: usize::MAX,
            mask: VertexBitSet::ZERO,
        }
    }

    /// `u`: Query vertex where this nogood is attached
    pub fn encode(mut k: Q, u: Vertex, assignment_ages: &Vec<usize>) -> Self {
        k.set(u as usize, false);

        if let Some(second_u) = k.last_one() {
            Nogood {
                u: second_u as Vertex,
                age: assignment_ages[second_u],
                mask: k,
            }
        } else {
            Nogood::permanent()
        }
    }

    #[allow(unused)]
    pub fn is_permanent(&self) -> bool {
        self.u == 0 && self.age == usize::MAX && self.mask == VertexBitSet::ZERO
    }

    pub fn check(
        &self,
        assignment_ages: &Vec<usize>,
        u: Vertex,
    ) -> Result<(), Q> {
        if assignment_ages
            .get(self.u as usize)
            .map_or(false, |&a| a <= self.age)
        {
            Err(self.mask | Q::from_vertex(u))
        } else {
            Ok(())
        }
    }
}

struct BinaryDisplay<'a, T>(&'a T);

impl<'a, T> fmt::Debug for BinaryDisplay<'a, T>
where
    T: fmt::Binary,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:b}", self.0)
    }
}

impl<Q: VertexBitSet> fmt::Display for Nogood<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nogood")
            .field("u", &self.u)
            .field("age", &self.age)
            .field("mask", &BinaryDisplay(&self.mask.load_le::<u64>()))
            .finish()
    }
}

impl<Q: VertexBitSet> HeapSizeOf for Nogood<Q> {
    fn heap_size_of(&self) -> usize {
        0
    }
}

pub fn allocate_vertex_nogoods<Q>(cs: &CandSpace) -> Vec<Vec<Nogood<Q>>>
where
    Q: VertexBitSet,
{
    (0..query_vertex_count(cs))
        .map(|u| vec![Nogood::default(); cs[u].candidates.len()])
        .collect()
}

/// `cs` must be reordered so that `u` is in the 2-core iff `u < core_size`.
pub fn allocate_edge_nogoods<Q: VertexBitSet>(
    cs: &CandSpace,
    core_size: usize,
) -> Vec<Vec<(Vertex, Vec<Vec<Nogood<Q>>>)>> {
    (0..core_size as Vertex)
        .map(|u| {
            cs[u as usize]
                .neighbors
                .iter()
                .take_while(|&&(u_, _)| u_ < core_size as Vertex)
                .map(|&(u_, ref inbsof)| {
                    let vs = (0..cand_count(cs, u))
                        .map(|iv| vec![Nogood::default(); inbsof[iv].len()])
                        .collect();
                    (u_, vs)
                })
                .collect()
        })
        .collect()
}
