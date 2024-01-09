use itertools::Itertools;
use log::trace;
use std::{cmp::Reverse, collections::HashSet};

use crate::graph::Vertex;
use crate::utils::HeapSizeOf;

use super::bitset::VertexBitSet;
use super::cand_space::CandSpace;
use super::embedding::Embedding;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct CoverInfo {
    freq: usize,
    non_trivial: bool,
}

/// `resvs[u + 1..qsize]` must has been computed before the call.
fn compute_reservation<Q: VertexBitSet>(
    cs: &CandSpace,
    matchas: &Vec<Q>,
    size_limit: usize,
    bwsets: &Vec<Q>,
    resvs: &Vec<Vec<Option<Vec<Vertex>>>>,
    u: Vertex,
    iv: usize,
    univ: &mut Vec<(Vertex, Vertex)>,
    cands: &mut Vec<Vertex>,
    chosen: &mut HashSet<Vertex>,
    coverinfo: &mut Vec<CoverInfo>,
) -> Option<Vec<Vertex>> {
    // `reservee` is a data vertex, and `reserver` is a query vertex.
    let is_acceptable_as_reservation = |reservee: Vertex, reserver: Vertex| {
        (matchas[reservee as usize] & bwsets[reserver as usize]).any()
    };
    let domsize = |dom_: &Q| (*dom_ & bwsets[u as usize]).count_ones();
    let domsizewith = |dom_: &Q, v_| {
        ((*dom_ | matchas[v_ as usize]) & bwsets[u as usize]).count_ones()
    };
    let v = cs[u as usize].candidates[iv];

    cs[u as usize]
        .neighbors
        .iter()
        .rev()
        .take_while(|&&(u_, _)| u_ > u) // Forward neighbors
        .flat_map(|&(u_, ref inbsof)| {
            debug_assert!(coverinfo.iter().all(|x| x == &CoverInfo::default()));
            univ.clear();
            cands.clear();
            chosen.clear();

            let mut dom = Q::ZERO;

            // Insert vertices that must be in the reservation into `chosen`
            // If any matchable reservation cannot be found, return None
            for &iv_ in &inbsof[iv] {
                let v_ = cs[u_ as usize].candidates[iv_];
                let v_ok = is_acceptable_as_reservation(v_, u);
                let trivialresv = [v_];

                let resv = if let Some(ref resv) = resvs[u_ as usize][iv_] {
                    resv.as_slice()
                } else {
                    // If a forward neighbor `v_` lacks the reservation,
                    // use trivial reservation [v_]
                    &trivialresv
                };

                for &w in resv.iter().filter(|&&w| w != v) {
                    // If v_ is chosen, w is never chosen and
                    // `!v_ok && !wok` never holds because v_ok is true
                    if chosen.contains(&v_) {
                        break;
                    }

                    let wok = is_acceptable_as_reservation(w, u);
                    if v_ok && (v_ == w || !wok) {
                        chosen.insert(v_);
                        dom |= matchas[v_ as usize];
                    } else if !v_ok && wok {
                        chosen.insert(w);
                        dom |= matchas[w as usize];
                    }

                    if (!v_ok && !wok)
                        || chosen.len() > domsize(&dom)
                        || chosen.len() > size_limit
                    {
                        chosen.clear();
                        return None;
                    }
                }
            }

            // Construct `univ`, the universal set to be covered
            for &iv_ in &inbsof[iv] {
                let v_ = cs[u_ as usize].candidates[iv_];
                if let Some(ref resv) = resvs[u_ as usize][iv_] {
                    if !chosen.contains(&v_) {
                        for &w in resv.iter().filter(|&&w| w != v) {
                            if !chosen.contains(&w) {
                                univ.push((v_, w));
                            }
                        }
                    }
                }
            }

            // Count the frequency of each vertex
            for &(v_, w) in univ.as_slice() {
                if coverinfo[v_ as usize].freq == 0 {
                    cands.push(v_);
                }
                if coverinfo[w as usize].freq == 0 {
                    cands.push(w);
                }
                coverinfo[v_ as usize].freq += 1;
                coverinfo[w as usize].freq += 1;
                coverinfo[w as usize].non_trivial = true; // non-trivial
            }

            let mut ncand = cands.len();

            while chosen.len() < size_limit {
                ncand = itertools::partition(&mut cands[0..ncand], |&v_| {
                    coverinfo[v_ as usize].freq > 0
                        && chosen.len() + 1 <= domsizewith(&dom, v_)
                });
                if ncand == 0 {
                    break;
                }

                let r = *cands
                    .iter()
                    .take(ncand)
                    .max_by_key(|&&v_| &coverinfo[v_ as usize])
                    .unwrap();
                chosen.insert(r);

                // Update the frequency and the uncovered set
                univ.retain(|&(v_, w)| {
                    debug_assert_ne!(v_, w);
                    if v_ == r || w == r {
                        coverinfo[v_ as usize].freq -= 1;
                        coverinfo[w as usize].freq -= 1;
                        false
                    } else {
                        true
                    }
                });

                // r will be removed from cand in the next loop
                debug_assert!(coverinfo[r as usize].freq == 0);
            }

            // Clean up for the next loop
            for &v_ in cands.as_slice() {
                coverinfo[v_ as usize] = CoverInfo::default();
            }

            if univ.is_empty() {
                Some((u_, chosen.iter().cloned().sorted().collect::<Vec<_>>()))
            } else {
                None
            }
        })
        .min_by_key(|(u_, rs)| (rs.len(), Reverse(*u_))) // Take greater u_
        .map(|(_, rs)| rs)
}

fn validate_reservations<Q: VertexBitSet>(
    resvs: &Vec<Vec<Option<Vec<Vertex>>>>,
    cs: &CandSpace,
    matchas: &Vec<Q>,
    size_limit: usize,
    bwsets: &Vec<Q>,
) -> bool {
    let qsize = cs.len();

    // The length of arrays should be same as the candidate space
    assert_eq!(resvs.len(), cs.len());
    for u in 0..qsize {
        assert_eq!(resvs[u].len(), cs[u].candidates.len());
    }

    // Reservations should be trivial (= None) or matchable
    for u in 0..qsize {
        // If size_limit == 0, all the reservation guards must be `None`
        if size_limit == 0 {
            assert!(resvs[u].iter().all(|r| r == &None || r == &Some(vec![])));
            continue;
        }

        // Otherwise, the size of every reservation guard must meet the limit
        for r in resvs[u].iter().flatten() {
            assert!(r.len() <= size_limit);

            // Condition (i): for all v' in R(u_i, v), C^{-1}(v')[:i] != {}
            for &v in r {
                assert!((matchas[v as usize] & bwsets[u as usize]).any())
            }

            // Condition (ii): for all S <= R(u_i, v), |S| <= |C^{-1}(S)[:i]|.
            // However, it is expensive to check the condition for all possible
            // S during the generation of reservation guards, and thus the
            // condition may not be held for some S.
            // Hence, here we check only for S = R(u_i, v).
            let mut c = Q::ZERO;
            for &v in r {
                c |= matchas[v as usize];
            }
            c &= bwsets[u as usize];
            assert!(r.len() <= c.count_ones() as usize);
        }
    }

    true
}

/// Set of reservation guards for the corresponding CandidateSpace.
///
/// Letting `cs` be the corresponding CandidateSpace, `self.0[u][iv]` is a
/// reservation guard attached to candidate vertex `cs[u].candidates[iv]`.
pub struct ReservationGuards(Vec<Vec<Option<Vec<Vertex>>>>);

impl ReservationGuards {
    pub fn generate<Q: VertexBitSet>(
        cs: &CandSpace,    // Optimized ordering
        matchas: &Vec<Q>,  // Original ordering
        ord: &Vec<Vertex>, // Reordered ID -> Original ID
        size_limit: usize, // Maximum size of a reservation guard
    ) -> Self {
        let qsize = cs.len();

        // `cs` is reordered, but `matchas` is in the original ordering.
        // So, we need to revert a vertex ID to the original one.
        let bwsets = (0..qsize)
            .map(|u| Q::from_vertices((0..u).map(|u_| ord[u_])))
            .collect::<Vec<_>>();

        trace!("Start computing reservation guards");

        // Reuse the allocated memory
        let mut univ = vec![]; // Universal set
        let mut cands = vec![];

        let nvert = matchas.len();
        let mut chosen = HashSet::with_capacity(qsize);
        let mut coverinfo = vec![CoverInfo::default(); nvert];

        let mut resvs = vec![vec![]; qsize];
        for u in (0..qsize).rev() {
            let u = u as Vertex;
            resvs[u as usize] = (0..cs[u as usize].candidates.len())
                .into_iter()
                .map(|iv| {
                    compute_reservation(
                        cs,
                        matchas,
                        size_limit,
                        &bwsets,
                        &resvs,
                        u,
                        iv,
                        &mut univ,
                        &mut cands,
                        &mut chosen,
                        &mut coverinfo,
                    )
                })
                .collect();
        }

        debug_assert!(validate_reservations(
            &resvs, cs, matchas, size_limit, &bwsets
        ));

        ReservationGuards(resvs)
    }

    /// Returns `Ok(())` if `m` does not match; otherwise returns `Err(k)` where
    /// `k` is a deadend mask.
    pub fn check<Q: VertexBitSet>(
        &self,
        m: &Embedding,
        u: Vertex,
        iv: usize,
    ) -> Result<(), Q> {
        let resv = &self.0[u as usize][iv];
        if resv.is_none() {
            return Ok(());
        }

        let resv = resv.as_ref().unwrap();
        let mut k = Q::from_vertex(u); // `u` is always in the deadend mask
        for &v in resv {
            if let Some(u_) = m.invert(v) {
                k.set(u_ as usize, true);
            } else {
                return Ok(());
            }
        }

        Err(k)
    }

    /// Returns the total number of vertices in each reservation guard
    pub fn total_size(&self) -> usize {
        self.0
            .iter()
            .map(|rs| rs.iter().flatten().map(|r| r.len()).sum::<usize>())
            .sum::<usize>()
    }

    /// Returns the size of the largest reservation guard
    pub fn max_guard_size(&self) -> usize {
        self.0
            .iter()
            .map(|rs| rs.iter().flatten().map(|r| r.len()).max().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }
}

impl HeapSizeOf for ReservationGuards {
    fn heap_size_of(&self) -> usize {
        self.0.heap_size_of()
    }
}
