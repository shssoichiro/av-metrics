use super::super::util::*;
use super::convolution_edge_s;
use core::arch::x86_64::*;

pub(crate) fn convolution_f32_avx_s(
    filter: &[f32],
    src: &[f32],
    dst: &mut [f32],
    tmp: &mut [f32],
    width: usize,
    height: usize,
    step: usize,
) {
    let n = match filter.len() {
        17 | 9 | 5 | 3 => filter.len(),
        _ => 0,
    };
    unsafe {
        convolution_f32_avx_s_1d(n, filter, src, dst, tmp, width, height);
    }
}

#[target_feature(enable = "avx")]
unsafe fn convolution_f32_avx_s_1d(
    n: usize,
    filter: &[f32],
    src: &[f32],
    dst: &mut [f32],
    tmp: &mut [f32],
    width: usize,
    height: usize,
) {
    let radius = filter.len() / 2;
    let width_mod8 = floorn(width, 8);
    let tmp_stride = ceiln(width, 8);

    let i_vec_end = height - radius;
    let j_vec_end = width_mod8 - ceiln(radius + 1, 8);

    // Vertical pass.
    for i in 0..radius {
        for j in 0..width {
            tmp[i * tmp_stride + j] =
                convolution_edge_s(false, filter, src, width, height, width, i, j);
        }
    }
    for i in radius..i_vec_end {
        convolution_f32_avx_s_1d_v_scanline(
            n,
            filter,
            src[(i * width)..].as_ptr(),
            tmp[(i * tmp_stride)..].as_mut_ptr(),
            width,
            width_mod8,
        );
        for j in width_mod8..width {
            tmp[i * tmp_stride + j] =
                convolution_edge_s(false, filter, src, width, height, width, i, j);
        }
    }
    for i in i_vec_end..height {
        for j in 0..width {
            tmp[i * tmp_stride + j] =
                convolution_edge_s(false, filter, src, width, height, width, i, j);
        }
    }

    // Horizontal pass.
    for i in 0..height {
        for j in 0..radius {
            dst[i * width + j] =
                convolution_edge_s(true, filter, tmp, width, height, tmp_stride, i, j);
        }
        convolution_f32_avx_s_1d_h_scanline(
            n,
            filter,
            tmp[(i * tmp_stride)..].as_ptr(),
            dst[(i * width)..].as_mut_ptr(),
            j_vec_end,
        );
        for j in (j_vec_end + radius)..width {
            dst[i * width + j] =
                convolution_edge_s(true, filter, tmp, width, height, tmp_stride, i, j);
        }
    }
}

type ConvVScanlineFn = unsafe fn(&[f32], *const f32, *mut f32, usize, usize);

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_v_scanline(
    n: usize,
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    src_stride: usize,
    j_end: usize,
) {
    let func: ConvVScanlineFn = match n {
        5 => convolution_f32_avx_s_1d_v_scanline_5,
        9 => convolution_f32_avx_s_1d_v_scanline_9,
        17 => convolution_f32_avx_s_1d_v_scanline_17,
        _ => convolution_f32_avx_s_1d_v_scanline_default,
    };
    func(filter, src, dst, src_stride, j_end);
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_v_scanline_5(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    src_stride: usize,
    j_end: usize,
) {
    assert!(filter.len() == 5);

    // radius = 2
    let src = src.sub(2 * src_stride);

    let f0 = _mm256_broadcast_ss(&filter[0]);
    let f1 = _mm256_broadcast_ss(&filter[1]);
    let f2 = _mm256_broadcast_ss(&filter[2]);
    let f3 = _mm256_broadcast_ss(&filter[3]);
    let f4 = _mm256_broadcast_ss(&filter[4]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        g = _mm256_load_ps(src.add(0 * src_stride + j));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_load_ps(src.add(1 * src_stride + j));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_load_ps(src.add(2 * src_stride + j));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_load_ps(src.add(3 * src_stride + j));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_load_ps(src.add(4 * src_stride + j));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);

        _mm256_store_ps(dst.add(j), sum0);
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_v_scanline_9(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    src_stride: usize,
    j_end: usize,
) {
    assert!(filter.len() == 9);

    // radius = 4
    let src = src.sub(4 * src_stride);

    // Evaluate filter taps 0-8
    let f0 = _mm256_broadcast_ss(&filter[0]);
    let f1 = _mm256_broadcast_ss(&filter[1]);
    let f2 = _mm256_broadcast_ss(&filter[2]);
    let f3 = _mm256_broadcast_ss(&filter[3]);
    let f4 = _mm256_broadcast_ss(&filter[4]);
    let f5 = _mm256_broadcast_ss(&filter[5]);
    let f6 = _mm256_broadcast_ss(&filter[6]);
    let f7 = _mm256_broadcast_ss(&filter[7]);
    let f8 = _mm256_broadcast_ss(&filter[8]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        g = _mm256_load_ps(src.add(0 * src_stride + j));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_load_ps(src.add(1 * src_stride + j));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_load_ps(src.add(2 * src_stride + j));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_load_ps(src.add(3 * src_stride + j));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_load_ps(src.add(4 * src_stride + j));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        g = _mm256_load_ps(src.add(5 * src_stride + j));
        g = _mm256_mul_ps(f5, g);
        sum1 = _mm256_add_ps(sum1, g);

        g = _mm256_load_ps(src.add(6 * src_stride + j));
        g = _mm256_mul_ps(f6, g);
        sum2 = _mm256_add_ps(sum2, g);

        g = _mm256_load_ps(src.add(7 * src_stride + j));
        g = _mm256_mul_ps(f7, g);
        sum3 = _mm256_add_ps(sum3, g);

        g = _mm256_load_ps(src.add(8 * src_stride + j));
        g = _mm256_mul_ps(f8, g);
        sum0 = _mm256_add_ps(sum0, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);

        _mm256_store_ps(dst.add(j), sum0);
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_v_scanline_17(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    src_stride: usize,
    j_end: usize,
) {
    assert!(filter.len() == 17);

    // radius = 8
    let src = src.sub(8 * src_stride);

    // Evaluate filter taps 0-8
    let f0 = _mm256_broadcast_ss(&filter[0]);
    let f1 = _mm256_broadcast_ss(&filter[1]);
    let f2 = _mm256_broadcast_ss(&filter[2]);
    let f3 = _mm256_broadcast_ss(&filter[3]);
    let f4 = _mm256_broadcast_ss(&filter[4]);
    let f5 = _mm256_broadcast_ss(&filter[5]);
    let f6 = _mm256_broadcast_ss(&filter[6]);
    let f7 = _mm256_broadcast_ss(&filter[7]);
    let f8 = _mm256_broadcast_ss(&filter[8]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        g = _mm256_load_ps(src.add(0 * src_stride + j));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_load_ps(src.add(1 * src_stride + j));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_load_ps(src.add(2 * src_stride + j));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_load_ps(src.add(3 * src_stride + j));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_load_ps(src.add(4 * src_stride + j));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        g = _mm256_load_ps(src.add(5 * src_stride + j));
        g = _mm256_mul_ps(f5, g);
        sum1 = _mm256_add_ps(sum1, g);

        g = _mm256_load_ps(src.add(6 * src_stride + j));
        g = _mm256_mul_ps(f6, g);
        sum2 = _mm256_add_ps(sum2, g);

        g = _mm256_load_ps(src.add(7 * src_stride + j));
        g = _mm256_mul_ps(f7, g);
        sum3 = _mm256_add_ps(sum3, g);

        g = _mm256_load_ps(src.add(8 * src_stride + j));
        g = _mm256_mul_ps(f8, g);
        sum0 = _mm256_add_ps(sum0, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);

        _mm256_store_ps(dst.add(j), sum0);
    }

    // Evaluate filter taps 9-16
    let f0 = _mm256_broadcast_ss(&filter[9]);
    let f1 = _mm256_broadcast_ss(&filter[10]);
    let f2 = _mm256_broadcast_ss(&filter[11]);
    let f3 = _mm256_broadcast_ss(&filter[12]);
    let f4 = _mm256_broadcast_ss(&filter[13]);
    let f5 = _mm256_broadcast_ss(&filter[14]);
    let f6 = _mm256_broadcast_ss(&filter[15]);
    let f7 = _mm256_broadcast_ss(&filter[16]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        g = _mm256_load_ps(src.add(9 * src_stride + j));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_load_ps(src.add(10 * src_stride + j));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_load_ps(src.add(11 * src_stride + j));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_load_ps(src.add(12 * src_stride + j));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_load_ps(src.add(13 * src_stride + j));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        g = _mm256_load_ps(src.add(14 * src_stride + j));
        g = _mm256_mul_ps(f5, g);
        sum1 = _mm256_add_ps(sum1, g);

        g = _mm256_load_ps(src.add(15 * src_stride + j));
        g = _mm256_mul_ps(f6, g);
        sum2 = _mm256_add_ps(sum2, g);

        g = _mm256_load_ps(src.add(16 * src_stride + j));
        g = _mm256_mul_ps(f7, g);
        sum3 = _mm256_add_ps(sum3, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);

        sum0 = _mm256_add_ps(_mm256_load_ps(dst.add(j)), sum0);
        _mm256_store_ps(dst.add(j), sum0);
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_v_scanline_default(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    src_stride: usize,
    j_end: usize,
) {
    let radius = filter.len() / 2;
    let src = src.sub(radius * src_stride);

    for y in (0..filter.len()).step_by(9) {
        let mut f0 = _mm256_setzero_ps();
        let mut f1 = _mm256_setzero_ps();
        let mut f2 = _mm256_setzero_ps();
        let mut f3 = _mm256_setzero_ps();
        let mut f4 = _mm256_setzero_ps();
        let mut f5 = _mm256_setzero_ps();
        let mut f6 = _mm256_setzero_ps();
        let mut f7 = _mm256_setzero_ps();
        let mut f8 = _mm256_setzero_ps();

        match filter.len() - y {
            1 => {
                f0 = _mm256_broadcast_ss(&filter[y + 0]);
            }
            2 => {
                f1 = _mm256_broadcast_ss(&filter[y + 1]);
            }
            3 => {
                f2 = _mm256_broadcast_ss(&filter[y + 2]);
            }
            4 => {
                f3 = _mm256_broadcast_ss(&filter[y + 3]);
            }
            5 => {
                f4 = _mm256_broadcast_ss(&filter[y + 4]);
            }
            6 => {
                f5 = _mm256_broadcast_ss(&filter[y + 5]);
            }
            7 => {
                f6 = _mm256_broadcast_ss(&filter[y + 6]);
            }
            8 => {
                f7 = _mm256_broadcast_ss(&filter[y + 7]);
            }
            _ => {
                f8 = _mm256_broadcast_ss(&filter[y + 8]);
            }
        }

        for j in (0..j_end).step_by(8) {
            let mut accum = _mm256_setzero_ps();
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let mut g;

            match filter.len() - y {
                1 => {
                    g = _mm256_load_ps(src.add((y + 0) * src_stride + j));
                    g = _mm256_mul_ps(f0, g);
                    sum0 = _mm256_add_ps(sum0, g);
                }
                2 => {
                    g = _mm256_load_ps(src.add((y + 1) * src_stride + j));
                    g = _mm256_mul_ps(f1, g);
                    sum1 = _mm256_add_ps(sum1, g);
                }
                3 => {
                    g = _mm256_load_ps(src.add((y + 2) * src_stride + j));
                    g = _mm256_mul_ps(f2, g);
                    sum2 = _mm256_add_ps(sum2, g);
                }
                4 => {
                    g = _mm256_load_ps(src.add((y + 3) * src_stride + j));
                    g = _mm256_mul_ps(f3, g);
                    sum3 = _mm256_add_ps(sum3, g);
                }
                5 => {
                    g = _mm256_load_ps(src.add((y + 4) * src_stride + j));
                    g = _mm256_mul_ps(f4, g);
                    sum0 = _mm256_add_ps(sum0, g);
                }
                6 => {
                    g = _mm256_load_ps(src.add((y + 5) * src_stride + j));
                    sum1 = _mm256_mul_ps(f5, g);
                }
                7 => {
                    g = _mm256_load_ps(src.add((y + 6) * src_stride + j));
                    sum2 = _mm256_mul_ps(f6, g);
                }
                8 => {
                    g = _mm256_load_ps(src.add((y + 7) * src_stride + j));
                    sum3 = _mm256_mul_ps(f7, g);
                }
                _ => {
                    g = _mm256_load_ps(src.add((y + 8) * src_stride + j));
                    sum0 = _mm256_mul_ps(f8, g);
                }
            }

            sum0 = _mm256_add_ps(sum0, sum2);
            sum1 = _mm256_add_ps(sum1, sum3);

            sum0 = _mm256_add_ps(sum0, sum1);
            accum = _mm256_add_ps(accum, sum0);

            if y > 0 {
                accum = _mm256_add_ps(accum, _mm256_load_ps(dst.add(j)));
            }

            _mm256_store_ps(dst.add(j), accum);
        }
    }
}

type ConvHScanlineFn = unsafe fn(&[f32], *const f32, *mut f32, usize);

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_h_scanline(
    n: usize,
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    j_end: usize,
) {
    let func: ConvHScanlineFn = match n {
        5 => convolution_f32_avx_s_1d_h_scanline_5,
        9 => convolution_f32_avx_s_1d_h_scanline_9,
        17 => convolution_f32_avx_s_1d_h_scanline_17,
        _ => convolution_f32_avx_s_1d_h_scanline_default,
    };
    func(filter, src, dst, j_end);
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_h_scanline_5(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    j_end: usize,
) {
    assert!(filter.len() == 5);

    let f0 = _mm256_broadcast_ss(&filter[0]);
    let f1 = _mm256_broadcast_ss(&filter[1]);
    let f2 = _mm256_broadcast_ss(&filter[2]);
    let f3 = _mm256_broadcast_ss(&filter[3]);
    let f4 = _mm256_broadcast_ss(&filter[4]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        let mut accum = _mm256_setzero_ps();
        g = _mm256_loadu_ps(src.add(j + 0));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_loadu_ps(src.add(j + 1));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_loadu_ps(src.add(j + 2));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_loadu_ps(src.add(j + 3));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_loadu_ps(src.add(j + 4));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);
        accum = _mm256_add_ps(accum, sum0);

        // radius = 2
        _mm256_storeu_ps(dst.add(j + 2), accum);
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_h_scanline_9(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    j_end: usize,
) {
    assert!(filter.len() == 9);

    let f0 = _mm256_broadcast_ss(&filter[0]);
    let f1 = _mm256_broadcast_ss(&filter[1]);
    let f2 = _mm256_broadcast_ss(&filter[2]);
    let f3 = _mm256_broadcast_ss(&filter[3]);
    let f4 = _mm256_broadcast_ss(&filter[4]);
    let f5 = _mm256_broadcast_ss(&filter[5]);
    let f6 = _mm256_broadcast_ss(&filter[6]);
    let f7 = _mm256_broadcast_ss(&filter[7]);
    let f8 = _mm256_broadcast_ss(&filter[8]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        let mut accum = _mm256_setzero_ps();

        g = _mm256_loadu_ps(src.add(j + 0));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_loadu_ps(src.add(j + 1));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_loadu_ps(src.add(j + 2));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_loadu_ps(src.add(j + 3));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_loadu_ps(src.add(j + 4));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        g = _mm256_loadu_ps(src.add(j + 5));
        g = _mm256_mul_ps(f5, g);
        sum1 = _mm256_add_ps(sum1, g);

        g = _mm256_loadu_ps(src.add(j + 6));
        g = _mm256_mul_ps(f6, g);
        sum2 = _mm256_add_ps(sum2, g);

        g = _mm256_loadu_ps(src.add(j + 7));
        g = _mm256_mul_ps(f7, g);
        sum3 = _mm256_add_ps(sum3, g);

        g = _mm256_loadu_ps(src.add(j + 8));
        g = _mm256_mul_ps(f8, g);
        sum0 = _mm256_add_ps(sum0, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);
        accum = _mm256_add_ps(accum, sum0);

        // radius = 4
        _mm256_storeu_ps(dst.add(j + 4), accum);
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_h_scanline_17(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    j_end: usize,
) {
    assert!(filter.len() == 17);

    // Evaluate filter taps 0-8
    let f0 = _mm256_broadcast_ss(&filter[0]);
    let f1 = _mm256_broadcast_ss(&filter[1]);
    let f2 = _mm256_broadcast_ss(&filter[2]);
    let f3 = _mm256_broadcast_ss(&filter[3]);
    let f4 = _mm256_broadcast_ss(&filter[4]);
    let f5 = _mm256_broadcast_ss(&filter[5]);
    let f6 = _mm256_broadcast_ss(&filter[6]);
    let f7 = _mm256_broadcast_ss(&filter[7]);
    let f8 = _mm256_broadcast_ss(&filter[8]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        let mut accum = _mm256_setzero_ps();

        g = _mm256_loadu_ps(src.add(j + 0));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_loadu_ps(src.add(j + 1));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_loadu_ps(src.add(j + 2));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_loadu_ps(src.add(j + 3));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_loadu_ps(src.add(j + 4));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        g = _mm256_loadu_ps(src.add(j + 5));
        g = _mm256_mul_ps(f5, g);
        sum1 = _mm256_add_ps(sum1, g);

        g = _mm256_loadu_ps(src.add(j + 6));
        g = _mm256_mul_ps(f6, g);
        sum2 = _mm256_add_ps(sum2, g);

        g = _mm256_loadu_ps(src.add(j + 7));
        g = _mm256_mul_ps(f7, g);
        sum3 = _mm256_add_ps(sum3, g);

        g = _mm256_loadu_ps(src.add(j + 8));
        g = _mm256_mul_ps(f8, g);
        sum0 = _mm256_add_ps(sum0, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);
        accum = _mm256_add_ps(accum, sum0);

        // radius = 8
        _mm256_store_ps(dst.add(j + 8), accum);
    }

    // Evaluate filter taps 9-16
    let f0 = _mm256_broadcast_ss(&filter[9]);
    let f1 = _mm256_broadcast_ss(&filter[10]);
    let f2 = _mm256_broadcast_ss(&filter[11]);
    let f3 = _mm256_broadcast_ss(&filter[12]);
    let f4 = _mm256_broadcast_ss(&filter[13]);
    let f5 = _mm256_broadcast_ss(&filter[14]);
    let f6 = _mm256_broadcast_ss(&filter[15]);
    let f7 = _mm256_broadcast_ss(&filter[16]);

    for j in (0..j_end).step_by(8) {
        let mut g;
        let mut sum0;
        let mut sum1;
        let sum2;
        let sum3;

        g = _mm256_loadu_ps(src.add(j + 9));
        g = _mm256_mul_ps(f0, g);
        sum0 = g;

        g = _mm256_loadu_ps(src.add(j + 10));
        g = _mm256_mul_ps(f1, g);
        sum1 = g;

        g = _mm256_loadu_ps(src.add(j + 11));
        g = _mm256_mul_ps(f2, g);
        sum2 = g;

        g = _mm256_loadu_ps(src.add(j + 12));
        g = _mm256_mul_ps(f3, g);
        sum3 = g;

        g = _mm256_loadu_ps(src.add(j + 13));
        g = _mm256_mul_ps(f4, g);
        sum0 = _mm256_add_ps(sum0, g);

        g = _mm256_loadu_ps(src.add(j + 14));
        g = _mm256_mul_ps(f5, g);
        sum1 = _mm256_add_ps(sum1, g);

        g = _mm256_loadu_ps(src.add(j + 15));
        g = _mm256_mul_ps(f6, g);
        sum2 = _mm256_add_ps(sum2, g);

        g = _mm256_loadu_ps(src.add(j + 16));
        g = _mm256_mul_ps(f7, g);
        sum3 = _mm256_add_ps(sum3, g);

        sum0 = _mm256_add_ps(sum0, sum2);
        sum1 = _mm256_add_ps(sum1, sum3);

        sum0 = _mm256_add_ps(sum0, sum1);

        // radius = 8
        sum0 = _mm256_add_ps(_mm256_load_ps(dst.add(j + 8)), sum0);
        _mm256_store_ps(dst.add(j + 8), sum0);
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn convolution_f32_avx_s_1d_h_scanline_default(
    filter: &[f32],
    src: *const f32,
    dst: *mut f32,
    j_end: usize,
) {
    let radius = filter.len() / 2;

    for x in (0..filter.len()).step_by(9) {
        let mut f0 = _mm256_setzero_ps();
        let mut f1 = _mm256_setzero_ps();
        let mut f2 = _mm256_setzero_ps();
        let mut f3 = _mm256_setzero_ps();
        let mut f4 = _mm256_setzero_ps();
        let mut f5 = _mm256_setzero_ps();
        let mut f6 = _mm256_setzero_ps();
        let mut f7 = _mm256_setzero_ps();
        let mut f8 = _mm256_setzero_ps();

        match filter.len() - x {
            1 => {
                f0 = _mm256_broadcast_ss(&filter[x + 0]);
            }
            2 => {
                f1 = _mm256_broadcast_ss(&filter[x + 1]);
            }
            3 => {
                f2 = _mm256_broadcast_ss(&filter[x + 2]);
            }
            4 => {
                f3 = _mm256_broadcast_ss(&filter[x + 3]);
            }
            5 => {
                f4 = _mm256_broadcast_ss(&filter[x + 4]);
            }
            6 => {
                f5 = _mm256_broadcast_ss(&filter[x + 5]);
            }
            7 => {
                f6 = _mm256_broadcast_ss(&filter[x + 6]);
            }
            8 => {
                f7 = _mm256_broadcast_ss(&filter[x + 7]);
            }
            _ => {
                f8 = _mm256_broadcast_ss(&filter[x + 8]);
            }
        }

        for j in (0..j_end).step_by(8) {
            let mut accum = _mm256_setzero_ps();
            let mut sum0;
            let mut sum1;
            let mut sum2;
            let mut sum3;
            let mut g;

            sum0 = _mm256_setzero_ps();
            sum1 = _mm256_setzero_ps();
            sum2 = _mm256_setzero_ps();
            sum3 = _mm256_setzero_ps();

            match filter.len() - x {
                1 => {
                    g = _mm256_loadu_ps(src.add(j + x + 0));
                    g = _mm256_mul_ps(f0, g);
                    sum0 = _mm256_add_ps(sum0, g);
                }
                2 => {
                    g = _mm256_loadu_ps(src.add(j + x + 1));
                    g = _mm256_mul_ps(f1, g);
                    sum1 = _mm256_add_ps(sum1, g);
                }
                3 => {
                    g = _mm256_loadu_ps(src.add(j + x + 2));
                    g = _mm256_mul_ps(f2, g);
                    sum2 = _mm256_add_ps(sum2, g);
                }
                4 => {
                    g = _mm256_loadu_ps(src.add(j + x + 3));
                    g = _mm256_mul_ps(f3, g);
                    sum3 = _mm256_add_ps(sum3, g);
                }
                5 => {
                    g = _mm256_loadu_ps(src.add(j + x + 4));
                    g = _mm256_mul_ps(f4, g);
                    sum0 = _mm256_add_ps(sum0, g);
                }
                6 => {
                    g = _mm256_loadu_ps(src.add(j + x + 5));
                    sum1 = _mm256_mul_ps(f5, g);
                }
                7 => {
                    g = _mm256_loadu_ps(src.add(j + x + 6));
                    sum2 = _mm256_mul_ps(f6, g);
                }
                8 => {
                    g = _mm256_loadu_ps(src.add(j + x + 7));
                    sum3 = _mm256_mul_ps(f7, g);
                }
                _ => {
                    g = _mm256_loadu_ps(src.add(j + x + 8));
                    sum0 = _mm256_mul_ps(f8, g);
                }
            }

            sum0 = _mm256_add_ps(sum0, sum2);
            sum1 = _mm256_add_ps(sum1, sum3);

            sum0 = _mm256_add_ps(sum0, sum1);
            accum = _mm256_add_ps(accum, sum0);

            if x > 0 {
                accum = _mm256_add_ps(accum, _mm256_loadu_ps(dst.add(j + radius)));
            }

            _mm256_storeu_ps(dst.add(j + radius), accum);
        }
    }
}
