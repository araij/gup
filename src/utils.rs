use itertools::Itertools;
use std::cmp::{Eq, PartialOrd};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::marker::Copy;
use std::mem::size_of;
use std::ops::{AddAssign, Index, IndexMut, Range, RangeBounds};
use std::ops::{Bound::*, Deref};
use std::time::{Duration, Instant};

//------------------------------------------------------------------------------
//
// Traits
//
//------------------------------------------------------------------------------

/// Define a total order (`Ord`) for `PartialOrd` types
///
/// https://qiita.com/hatoo@github/items/fa14ad36a1b568d14f3e
#[derive(PartialEq, PartialOrd)]
pub struct Total<T>(pub T);

impl<T: Debug> Debug for Total<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_tuple("Total").field(&self.0).finish()
    }
}

impl<T: PartialEq> Eq for Total<T> {}

impl<T: PartialOrd> Ord for Total<T> {
    fn cmp(&self, other: &Total<T>) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl<T: Copy> Copy for Total<T> {}

impl<T: Clone> Clone for Total<T> {
    fn clone(&self) -> Self {
        Total(self.0.clone())
    }
}

pub trait IsOrdered: Iterator {
    /// Equivalent of `Iterator::is_sorted()` in the nightly build.
    ///
    /// Replace with `std`'s implementation when it become stable.
    /// Note that renaming this with `is_sorted` raises a warning of
    /// "unstable_name_collisions". This can be suppressed only by callers.
    fn is_ordered(self) -> bool
    where
        Self: Sized,
        Self::Item: Clone + Ord,
    {
        self.tuple_windows().all(|(x, y)| x <= y)
    }
}

impl<I: Iterator> IsOrdered for I {}

pub trait HeapSizeOf {
    /// Returns heap memory consumption in bytes
    fn heap_size_of(&self) -> usize;
}

impl HeapSizeOf for u32 {
    fn heap_size_of(&self) -> usize {
        0
    }
}

impl<T0, T1> HeapSizeOf for (T0, T1)
where
    T0: HeapSizeOf,
    T1: HeapSizeOf,
{
    fn heap_size_of(&self) -> usize {
        self.0.heap_size_of() + self.1.heap_size_of()
    }
}

impl<T> HeapSizeOf for Option<T>
where
    T: HeapSizeOf,
{
    fn heap_size_of(&self) -> usize {
        self.as_ref().map_or(0, |x| x.heap_size_of())
    }
}

impl<T> HeapSizeOf for Vec<T>
where
    T: HeapSizeOf,
{
    fn heap_size_of(&self) -> usize {
        self.capacity() * size_of::<T>()
            + self.iter().map(|x| x.heap_size_of()).sum::<usize>()
    }
}

//------------------------------------------------------------------------------
//
// Iterator utilities
//
//------------------------------------------------------------------------------

/// Returns items that appears in both `xs` and `ys`. Both must be sorted.
#[allow(dead_code)]
pub fn intersect<I, J>(xs: I, ys: J) -> impl Iterator<Item = I::Item>
where
    I: IntoIterator,
    I::Item: Clone + Ord,
    J: IntoIterator<Item = I::Item>,
{
    itertools::merge_join_by(xs, ys, |x, y| x.cmp(y))
        .flat_map(|e| e.both())
        .map(|(x, _)| x)
}

#[allow(dead_code)]
fn partial_sum<T, I>(xs: I) -> impl Iterator<Item = T>
where
    T: AddAssign + Clone,
    I: IntoIterator<Item = T>,
{
    xs.into_iter().scan(None, |sum, i| {
        match sum {
            None => *sum = Some(i),
            Some(x) => *x += i,
        }
        sum.clone()
    })
}

//------------------------------------------------------------------------------
//
// Collection types
//
//------------------------------------------------------------------------------

/// A vector that is split into some sections.
///
/// All the sections are stored in a single contiguous memory region.
/// Each section is accessible with a indexing operator.
pub struct SplitVec<T> {
    starts: Vec<usize>,
    data: Vec<T>,
}

impl<T> Index<usize> for SplitVec<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &[T] {
        assert!(index < self.len());
        &self.data[self.starts[index]..self.starts[index + 1]]
    }
}

impl<T> IndexMut<usize> for SplitVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut [T] {
        assert!(index < self.len());
        &mut self.data[self.starts[index]..self.starts[index + 1]]
    }
}

impl<T> SplitVec<T> {
    pub fn new() -> SplitVec<T> {
        SplitVec {
            starts: vec![0],
            data: vec![],
        }
    }

    pub fn with_capacity(section: usize, data: usize) -> SplitVec<T> {
        let mut x = SplitVec::new();
        x.reserve(section, data);
        x
    }

    pub fn len(&self) -> usize {
        self.starts.len() - 1
    }

    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    pub fn push(&mut self, x: T) {
        self.data.push(x)
    }

    pub fn close(&mut self) {
        self.starts.push(self.data.len())
    }

    pub fn reserve(&mut self, section: usize, data: usize) {
        self.starts.reserve(section + 1);
        self.data.reserve(data);
    }

    pub fn shrink_to_fit(&mut self) {
        self.starts.shrink_to_fit();
        self.data.shrink_to_fit();
    }
}

//------------------------------------------------------------------------------
//
// Misc.
//
//------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Timer {
    duration: Duration,
    start: Instant,
}

impl Timer {
    pub fn started(duration: Duration) -> Self {
        Timer {
            duration,
            start: Instant::now(),
        }
    }

    pub fn is_over(&self) -> bool {
        self.duration <= self.start.elapsed()
    }
}

pub struct RevIndex<T>(pub T)
where
    T: Deref + AsRef<[<<T as Deref>::Target as Index<usize>>::Output]>,
    <T as Deref>::Target: Index<usize>,
    <<T as Deref>::Target as Index<usize>>::Output: Sized;

impl<T> RevIndex<T>
where
    T: Deref
        + AsRef<[<<T as Deref>::Target as Index<usize>>::Output]>
        + AsMut<[<<T as Deref>::Target as Index<usize>>::Output]>,
    <T as Deref>::Target: Index<usize>,
    <<T as Deref>::Target as Index<usize>>::Output: Sized,
{
    #[allow(dead_code)]
    pub fn get(
        &self,
        index: usize,
    ) -> Option<&<<T as Deref>::Target as Index<usize>>::Output> {
        if self.0.as_ref().len() > index {
            Some(&self[index])
        } else {
            None
        }
    }

    pub fn get_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut <<T as Deref>::Target as Index<usize>>::Output> {
        if self.0.as_ref().len() > index {
            Some(&mut self[index])
        } else {
            None
        }
    }
}

impl<T> Index<usize> for RevIndex<T>
where
    T: Deref + AsRef<[<<T as Deref>::Target as Index<usize>>::Output]>,
    <T as Deref>::Target: Index<usize>,
    <<T as Deref>::Target as Index<usize>>::Output: Sized,
{
    type Output = <<T as Deref>::Target as Index<usize>>::Output;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0.as_ref()[self.0.as_ref().len() - 1 - index]
    }
}

impl<T> IndexMut<usize> for RevIndex<T>
where
    T: Deref
        + AsRef<[<<T as Deref>::Target as Index<usize>>::Output]>
        + AsMut<[<<T as Deref>::Target as Index<usize>>::Output]>,
    <T as Deref>::Target: Index<usize>,
    <<T as Deref>::Target as Index<usize>>::Output: Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i = self.0.as_ref().len() - 1 - index;
        &mut self.0.as_mut()[i]
    }
}

/// Convert `impl RangeBounds<usize>` into `Range<usize>`
pub fn to_range<R: RangeBounds<usize>>(
    r: R,
    unbounded_end: usize,
) -> Range<usize> {
    let lo = match r.start_bound() {
        Included(&i) => i,
        Excluded(&i) => i + 1,
        Unbounded => 0,
    };
    let hi = match r.end_bound() {
        Included(&i) => i + 1,
        Excluded(&i) => i,
        Unbounded => unbounded_end,
    };
    lo..hi
}

//------------------------------------------------------------------------------
//
// Tests
//
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect() {
        itertools::assert_equal(
            intersect((0..=12).step_by(2), (0..=12).step_by(3)),
            vec![0, 6, 12],
        );

        let empty: Vec<i32> = vec![];
        itertools::assert_equal(intersect(&empty, &empty), &empty);
        itertools::assert_equal(intersect(&vec![0], &empty), &empty);
    }

    #[test]
    fn test_partial_sum() {
        assert_eq!(partial_sum::<i32, _>([]).count(), 0);
        itertools::assert_equal(partial_sum(0..6), [0, 1, 3, 6, 10, 15]);
    }
}
