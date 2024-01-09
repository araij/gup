use crate::graph::Vertex;
use crate::utils::*;
use bitvec::field::BitField;
use bitvec::macros::internal::funty;
use bitvec::prelude::*;
use bitvec::slice::IterOnes;
use derive_more::*;
use std::fmt::Debug;
use std::ops::*;

////////////////////////////////////////////////////////////////////////////////
//
// VertexBitSet
//
////////////////////////////////////////////////////////////////////////////////

/// Bit-set of query vertices.
///
/// The default backing storage of `BitArray` is `usize`, but its size may
/// differ depending on platforms. For simplicity, we assume that backing
/// storage is always 64 bits.
pub trait VertexBitSet:
    BitAnd<Output = Self>
    + BitAndAssign
    + for<'a> BitAndAssign<&'a Self>
    + BitOr<Output = Self>
    + BitOrAssign
    + Clone
    + Copy
    + Debug
    + Default
    + Eq
    + Index<usize, Output = bool>
    + Not<Output = Self>
    + Sized
    + Send
{
    const ZERO: Self;

    fn from_vertex(u: Vertex) -> Self;
    fn from_vertices<I: IntoIterator<Item = Vertex>>(us: I) -> Self;
    fn clear_after(&mut self, index: usize);

    // Following functions are derived from `BitSlice`
    fn any(&self) -> bool;
    fn count_ones(&self) -> usize;
    fn iter_ones(&self) -> IterOnes<u64, Lsb0>;
    fn last_one(&self) -> Option<usize>;
    fn len(&self) -> usize;
    fn load_le<I: funty::Integral>(&self) -> I;
    fn not_any(&self) -> bool;
    fn set(&mut self, index: usize, value: bool);
    fn shift_left(&mut self, by: usize);
    fn shift_right(&mut self, by: usize);
}

impl<const N: usize> VertexBitSet for BitArray<[u64; N]> {
    const ZERO: Self = BitArray::<[u64; N]>::ZERO;

    fn from_vertex(u: Vertex) -> Self {
        let mut b = Self::ZERO;
        b.set(u as usize, true);
        b
    }

    fn from_vertices<I: IntoIterator<Item = Vertex>>(us: I) -> Self {
        let mut b = Self::ZERO;
        for u in us {
            b.set(u as usize, true);
        }
        b
    }

    fn clear_after(&mut self, index: usize) {
        self.fill_range(index.., false)
    }

    fn any(&self) -> bool {
        self.as_bitslice().any()
    }

    fn count_ones(&self) -> usize {
        self.as_bitslice().count_ones()
    }

    fn iter_ones(&self) -> IterOnes<u64, Lsb0> {
        self.as_bitslice().iter_ones()
    }

    fn last_one(&self) -> Option<usize> {
        self.as_bitslice().last_one()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn load_le<I: funty::Integral>(&self) -> I {
        self.as_bitslice().load_le()
    }

    fn not_any(&self) -> bool {
        self.as_bitslice().not_any()
    }

    fn set(&mut self, index: usize, value: bool) {
        self.as_mut_bitslice().set(index, value)
    }

    fn shift_left(&mut self, by: usize) {
        self.as_mut_bitslice().shift_left(by)
    }

    fn shift_right(&mut self, by: usize) {
        self.as_mut_bitslice().shift_right(by)
    }
}

pub type VertexBitSet128 = BitArray<[u64; 2]>;
pub type VertexBitSet256 = BitArray<[u64; 4]>;

////////////////////////////////////////////////////////////////////////////////
//
// VertexBitSet64
//
////////////////////////////////////////////////////////////////////////////////

/// Implementation of VertexBitSet for up to 64 vertices
#[derive(
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Not,
    PartialEq,
)]
pub struct VertexBitSet64(u64);

impl VertexBitSet for VertexBitSet64 {
    const ZERO: Self = VertexBitSet64(0);

    fn from_vertex(u: Vertex) -> Self {
        VertexBitSet64((1 as u64) << u)
    }

    fn from_vertices<I: IntoIterator<Item = Vertex>>(us: I) -> Self {
        let mut b = 0;
        for u in us {
            b |= (1 as u64) << u;
        }
        VertexBitSet64(b)
    }

    fn clear_after(&mut self, index: usize) {
        // Somehow shifts of BitArray are expensive, and so we touch raw data
        let s = 64 - index;
        self.0 = (self.0 << s) >> s;
    }

    fn any(&self) -> bool {
        self.0 != 0
    }

    fn count_ones(&self) -> usize {
        self.0.count_ones() as usize
    }

    fn iter_ones(&self) -> IterOnes<u64, Lsb0> {
        self.0.view_bits::<Lsb0>().iter_ones()
    }

    fn last_one(&self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(63 - self.0.leading_zeros() as usize)
        }
    }

    fn len(&self) -> usize {
        64
    }

    fn load_le<I: funty::Integral>(&self) -> I {
        self.0.view_bits::<Lsb0>().load_le()
    }

    fn not_any(&self) -> bool {
        self.0 == 0
    }

    fn set(&mut self, index: usize, value: bool) {
        let b = (1 as u64) << index;
        if value {
            self.0 |= b;
        } else {
            self.0 &= !b;
        }
    }

    fn shift_left(&mut self, by: usize) {
        self.0 = self.0 >> by
    }

    fn shift_right(&mut self, by: usize) {
        self.0 = self.0 << by
    }
}

impl Index<usize> for VertexBitSet64 {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        if (self.0 & ((1 as u64) << index)) == 0 {
            &false
        } else {
            &true
        }
    }
}

impl BitAndAssign<&VertexBitSet64> for VertexBitSet64 {
    fn bitand_assign(&mut self, rhs: &VertexBitSet64) {
        self.0 &= rhs.0;
    }
}

pub trait FillRange {
    fn fill_range<R: RangeBounds<usize>>(&mut self, r: R, v: bool);
}

impl<Q> FillRange for Q
where
    Q: VertexBitSet,
{
    fn fill_range<R: RangeBounds<usize>>(&mut self, r: R, v: bool) {
        let Range { start: lo, end: hi } = to_range(r, self.len());
        if lo < hi {
            let mut m = !Q::ZERO;
            m.shift_left(self.len() - (hi - lo));
            m.shift_right(lo);
            if v {
                *self |= m;
            } else {
                *self &= !m;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Public Free Functions
//
////////////////////////////////////////////////////////////////////////////////

pub fn is_superset<Q: VertexBitSet>(b: &Q, other: &Q) -> bool {
    let mut b = b.clone();
    b &= other;
    b == *other
}

#[allow(dead_code)]
pub fn iter_vertices<'a, Q: VertexBitSet>(
    b: &'a Q,
) -> impl DoubleEndedIterator<Item = Vertex> + 'a {
    b.iter_ones().map(|x| x as Vertex)
}

pub fn last_vertex<Q: VertexBitSet>(b: &Q) -> Option<Vertex> {
    b.last_one().map(|x| x as Vertex)
}

////////////////////////////////////////////////////////////////////////////////
//
// Tests
//
////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_set_range() {
    let mut b = VertexBitSet64::ZERO;

    // 0000000000000000000...
    b.fill_range(16..8, true); // nothing changed
    assert!(b.not_any());

    // 0000000011111111111...
    b.fill_range(8.., true);
    dbg!(b.0);
    assert!(!(0..8).any(|i| b[i]));
    assert!((8..b.len()).all(|i| b[i]));

    // 0000000000000000000...
    b.fill_range(..b.len(), false);
    assert!(!(0..b.len()).any(|i| b[i]));

    // 0000000011111111000...
    b.fill_range(8..16, true);
    assert!(!(0..8).any(|i| b[i]));
    assert!((8..16).all(|i| b[i]));
    assert!(!(16..b.len()).any(|i| b[i]));

    // 0000000000001111000...
    b.fill_range(4..12, false);
    assert!(!(0..12).any(|i| b[i]));
    assert!((12..16).all(|i| b[i]));
    assert!(!(16..b.len()).any(|i| b[i]));

    // 1111000000001111000...
    b.fill_range(0..4, true);
    assert!((0..4).all(|i| b[i]));
    assert!(!(4..12).any(|i| b[i]));
    assert!((12..16).all(|i| b[i]));
    assert!(!(16..b.len()).any(|i| b[i]));
}
