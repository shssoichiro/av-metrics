use crate::video::CastFromPrimitive;
use crate::video::FrameInfo;
use crate::video::Pixel;
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

pub(crate) fn import_frame_data<T: Pixel>(frame: &FrameInfo<T>) -> Vec<f32> {
    // VMAF only uses the Y plane
    frame.planes[0]
        .data
        .iter()
        .copied()
        .map(|pix| u16::cast_from(pix) as f32)
        .collect()
}

pub(crate) fn offset_image(img_buf: &mut [f32], offset: f32) {
    for pix in img_buf.iter_mut() {
        *pix += offset;
    }
}

pub(crate) fn floorn(n: usize, m: usize) -> usize {
    n - n % m
}

pub(crate) fn ceiln(n: usize, m: usize) -> usize {
    if n % m > 0 {
        n + (m - n % m)
    } else {
        n
    }
}

pub(crate) const MAX_ALIGN: usize = 32;

#[inline(always)]
pub(crate) fn align_floor(x: usize) -> usize {
    x - x % MAX_ALIGN
}

#[inline(always)]
pub(crate) fn align_ceil(x: usize) -> usize {
    x + if x % MAX_ALIGN > 0 {
        MAX_ALIGN - x % MAX_ALIGN
    } else {
        0
    }
}

#[repr(align(32))]
pub(crate) struct Align32;

// A 32 byte aligned array.
// # Examples
// ```
// let mut x: AlignedArray<[i32; 64 * 64]> = AlignedArray::new([0; 64 * 64]);
// assert!(x.array.as_ptr() as usize % 32 == 0);
//
// let mut x: AlignedArray<[i32; 64 * 64]> = AlignedArray::uninitialized();
// assert!(x.array.as_ptr() as usize % 32 == 0);
// ```
pub(crate) struct AlignedArray<ARRAY> {
    _alignment: [Align32; 0],
    pub array: ARRAY,
}

impl<A> AlignedArray<A> {
    pub(crate) const fn new(array: A) -> Self {
        AlignedArray {
            _alignment: [],
            array,
        }
    }
    pub(crate) fn uninitialized() -> Self {
        Self::new(unsafe { MaybeUninit::uninit().assume_init() })
    }
}

// A 32 byte aligned vector.
pub(crate) struct AlignedVec<T> {
    _alignment: [Align32; 0],
    pub vec: Vec<T>,
}

impl<T> AlignedVec<T> {
    pub(crate) const fn new(vec: Vec<T>) -> Self {
        AlignedVec {
            _alignment: [],
            vec,
        }
    }
    pub(crate) fn uninitialized() -> Self {
        Self::new(unsafe { MaybeUninit::uninit().assume_init() })
    }
}

impl<T> Index<usize> for AlignedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl<T> IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vec[index]
    }
}
