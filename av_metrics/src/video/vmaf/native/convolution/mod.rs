use super::util::*;

mod avx;

use avx::*;

// A lot of the C codebase is designed with the idea that
// there could be other filters in the future,
// but currently this is the only one that exists.
//
// The Rust codebase will code for future filters,
// and allow the compiler to optimize away any logic
// based on this const slice.
#[allow(clippy::excessive_precision)]
pub(crate) const FILTER: [f32; 5] = [
    0.054488685,
    0.244201342,
    0.402619947,
    0.244201342,
    0.054488685,
];

pub(crate) fn convolution_f32_c_s(
    filter: &[f32],
    src: &[f32],
    dst: &mut [f32],
    tmp: &mut [f32],
    width: usize,
    height: usize,
    use_simd: bool,
) {
    if is_x86_feature_detected!("avx2") && use_simd {
        convolution_f32_avx_s(filter, src, dst, tmp, width, height, 1);
        return;
    }

    convolution_y_c_s(filter, src, tmp, width, height, 1);
    convolution_x_c_s(filter, tmp, dst, width, height, 1);
}

fn convolution_y_c_s(
    filter: &[f32],
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    step: usize,
) {
    let filter_width = filter.len();
    let radius = filter_width / 2;
    let borders_top = ceiln(radius, step);
    let borders_bottom = floorn(height - (filter_width - radius), step);

    for i in (0..borders_top).step_by(step) {
        for j in 0..width {
            dst[(i / step) * width + j] =
                convolution_edge_s(false, filter, src, width, height, width, i, j);
        }
    }

    for i in (borders_top..borders_bottom).step_by(step) {
        for j in 0..width {
            let mut accum = 0.;
            for k in 0..filter_width {
                accum += filter[k] * src[(i - radius + k) * width + j];
            }
            dst[(i / step) * width + j] = accum;
        }
    }

    for i in (borders_bottom..height).step_by(step) {
        for j in 0..width {
            dst[(i / step) * width + j] =
                convolution_edge_s(false, filter, src, width, height, width, i, j);
        }
    }
}

fn convolution_x_c_s(
    filter: &[f32],
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    step: usize,
) {
    let filter_width = filter.len();
    let radius = filter_width / 2;
    let borders_left = ceiln(radius, step);
    let borders_right = floorn(width - (filter_width - radius), step);

    for i in 0..height {
        for j in (0..borders_left).step_by(step) {
            dst[i * width + j / step] =
                convolution_edge_s(true, filter, src, width, height, width, i, j);
        }

        for j in (borders_left..borders_right).step_by(step) {
            let mut accum = 0.;
            for k in 0..filter_width {
                accum += filter[k] * src[i * width + j - radius + k];
            }
            dst[i * width + j / step] = accum;
        }

        for j in (borders_right..width).step_by(step) {
            dst[i * width + j / step] =
                convolution_edge_s(true, filter, src, width, height, width, i, j);
        }
    }
}

#[inline(always)]
fn convolution_edge_s(
    horizontal: bool,
    filter: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    i: usize,
    j: usize,
) -> f32 {
    let filter_width = filter.len();
    let radius = filter_width / 2;

    let mut accum = 0.;
    for k in 0..filter_width {
        let mut i_tap = if horizontal {
            i as isize
        } else {
            i as isize - radius as isize + k as isize
        };
        let mut j_tap = if horizontal {
            j as isize - radius as isize + k as isize
        } else {
            j as isize
        };

        // Handle edges by mirroring.
        if horizontal {
            if j_tap < 0 {
                j_tap = -j_tap;
            } else if j_tap as usize >= width {
                j_tap = width as isize - (j_tap - width as isize + 1);
            }
        } else if i_tap < 0 {
            i_tap = -i_tap;
        } else if i_tap as usize >= height {
            i_tap = height as isize - (i_tap - height as isize + 1);
        }

        accum += filter[k] * src[i_tap as usize * stride + j_tap as usize];
    }
    accum
}
