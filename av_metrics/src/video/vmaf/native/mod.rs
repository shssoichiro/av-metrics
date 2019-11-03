use self::convolution::FILTER;
use self::convolution::*;
use self::util::*;
use super::models;
use crate::video::*;
use std::error::Error;
use std::mem::size_of;

mod convolution;
mod util;

#[derive(Clone)]
pub(crate) struct VmafMetrics {
    pub(crate) score: f64,
    blur_buf: Vec<f32>,
}

pub(crate) fn compute_vmaf<T: Pixel>(
    frame1: &FrameInfo<T>,
    frame2: &FrameInfo<T>,
    previous_metrics: Option<&VmafMetrics>,
    use_simd: bool,
) -> Result<VmafMetrics, Box<dyn Error>> {
    let width = frame1.planes[0].width;
    let height = frame1.planes[0].height;
    let (model, svm) = if width >= 2880 {
        // This is somewhat of an arbitrary cutoff,
        // but it attempts to handle videos between 1080p and 4K
        // in a reasonable manner.
        (models::get_vmaf_4k_model(), models::get_vmaf_4k_svm())
    } else {
        (models::get_vmaf_def_model(), models::get_vmaf_def_svm())
    };

    let psnr = psnr::calculate_frame_psnr(frame1, frame2)?;
    let ssim = ssim::calculate_frame_ssim(frame1, frame2)?;
    let msssim = ssim::calculate_frame_msssim(frame1, frame2)?;

    let mut ref_buf = import_frame_data(frame1);
    let mut dis_buf = import_frame_data(frame2);
    let mut blur_buf = vec![0f32; ref_buf.len()];
    let mut tmp_buf = vec![0f32; ref_buf.len()];

    offset_image(&mut ref_buf, -128.);
    offset_image(&mut dis_buf, -128.);

    convolution_f32_c_s(
        &FILTER,
        &ref_buf,
        &mut blur_buf,
        &mut tmp_buf,
        width,
        height,
        use_simd,
    );
    let adm = compute_adm(&ref_buf, &dis_buf, width, height);
    compute_motion();
    compute_vif();

    // TODO
    unimplemented!()
}

#[derive(Clone, Copy)]
struct AdmResults {
    score: f64,
    score_num: f64,
    score_den: f64,
    scores: [f64; 8],
}

fn compute_adm(
    ref_buf: &[f32],
    dis_buf: &[f32],
    width: usize,
    height: usize,
) -> Result<AdmResults, String> {
    let numden_limit = 1e-10 * (width * height) as f64 / (1920u64 * 1080u64) as f64;

    let curr_ref_scale = &ref_buf;
    let curr_dis_scale = &dis_buf;
    let mut curr_ref_stride = width;
    let mut curr_dis_stride = width;
    let orig_h = height;

    let mut ind_y: [&mut [i32]; 4] = [&mut []; 4];
    let mut ind_x: [&mut [i32]; 4] = [&mut []; 4];

    let buf_stride = align_ceil(((width + 1) / 2) * size_of::<f32>());
    let buf_sz_one = buf_stride * ((height + 1) / 2);
    let ind_size_y = align_ceil(((height + 1) / 2) * size_of::<i32>());
    let ind_size_x = align_ceil(((width + 1) / 2) * size_of::<i32>());

    let mut num = 0f64;
    let mut den = 0f64;

    // Code optimized to save on multiple buffer copies
    // hence the reduction in the number of buffers required from 35 to 17
    const NUM_BUFS_ADM: usize = 20;

    let mut data_buf = AlignedArray::new([0f32; NUM_BUFS_ADM]);

    let data_top = &data_buf;
    let (ref_dwt2, data_top) = init_dwt_band(data_top, buf_sz_one);
    let (dis_dwt2, data_top) = init_dwt_band(data_top, buf_sz_one);
    let (decouple_r, data_top) = init_dwt_band_hvd(data_top, buf_sz_one);
    let (decouple_a, data_top) = init_dwt_band_hvd(data_top, buf_sz_one);
    let (csf_a, data_top) = init_dwt_band_hvd(data_top, buf_sz_one);
    let (csf_f, data_top) = init_dwt_band_hvd(data_top, buf_sz_one);

    let mut buf_y_orig = AlignedVec::new(vec![[0i32; 4]; ind_size_y]);
    ind_y[0] = &mut buf_y_orig[ind_size_y * 0];
    ind_y[1] = &mut buf_y_orig[ind_size_y * 1];
    ind_y[2] = &mut buf_y_orig[ind_size_y * 2];
    ind_y[3] = &mut buf_y_orig[ind_size_y * 3];

    // if (!(buf_x_orig = aligned_malloc(ind_size_x * 4, MAX_ALIGN)))
    // {
    // 	printf("error: aligned_malloc failed for ind_buf_x.\n");
    // 	fflush(stdout);
    // 	goto fail;
    // }
    // ind_buf_x = buf_x_orig;
    // ind_x[0] = (int*)ind_buf_x; ind_buf_x += ind_size_x;
    // ind_x[1] = (int*)ind_buf_x; ind_buf_x += ind_size_x;
    // ind_x[2] = (int*)ind_buf_x; ind_buf_x += ind_size_x;
    // ind_x[3] = (int*)ind_buf_x; ind_buf_x += ind_size_x;

    // for (scale = 0; scale < 4; ++scale) {
    // float num_scale = 0.0;
    // float den_scale = 0.0;

    // dwt2_src_indices_filt(ind_y, ind_x, w, h);
    // adm_dwt2(curr_ref_scale, &ref_dwt2, ind_y, ind_x, w, h, curr_ref_stride, buf_stride);
    // adm_dwt2(curr_dis_scale, &dis_dwt2, ind_y, ind_x, w, h, curr_dis_stride, buf_stride);

    // w = (w + 1) / 2;
    // h = (h + 1) / 2;

    // adm_decouple(&ref_dwt2, &dis_dwt2, &decouple_r, &decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride, border_factor);

    // den_scale = adm_csf_den_scale(&ref_dwt2, orig_h, scale, w, h, buf_stride, border_factor);

    // adm_csf(&decouple_a, &csf_a, &csf_f, orig_h, scale, w, h, buf_stride, buf_stride, border_factor);

    // num_scale = adm_cm(&decouple_r, &csf_f, &csf_a, w, h, buf_stride, buf_stride, buf_stride, border_factor, scale);

    // num += num_scale;
    // den += den_scale;

    // ref_scale = ref_dwt2.band_a;
    // dis_scale = dis_dwt2.band_a;

    // curr_ref_scale = ref_scale;
    // curr_dis_scale = dis_scale;

    // curr_ref_stride = buf_stride;
    // curr_dis_stride = buf_stride;

    // scores[2 * scale + 0] = num_scale;
    // 	scores[2 * scale + 1] = den_scale;
    // }

    // num = num < numden_limit ? 0 : num;
    // den = den < numden_limit ? 0 : den;

    // if (den == 0.0)
    // {
    // 	*score = 1.0f;
    // }
    // else
    // {
    // 	*score = num / den;
    // }
    // *score_num = num;
    // *score_den = den;

    // ret = 0;
}

fn compute_motion() {
    // // compute
    //            if (frm_idx == 0)
    //            {
    //                score = 0.0;
    //                score2 = 0.0;
    //            }
    //            else
    //            {
    //                // avoid multiple memory copies
    //                prev_blur_buf = get_blur_buf(&thread_data->blur_buf_array, frm_idx - 1);
    //                if(NULL == prev_blur_buf)
    //                {
    //                    thread_data->stop_threads = 1;
    //                    sprintf(errmsg, "Data not available for prev_blur_buf.\n");
    //                    goto fail_or_end;
    //                }
    //                if ((ret = compute_motion(prev_blur_buf, blur_buf, w, h, stride, stride, &score)))
    //                {
    //                    sprintf(errmsg, "compute_motion (prev) failed.\n");
    //                    goto fail_or_end;
    //                }
    //                release_blur_buf_reference(&thread_data->blur_buf_array, frm_idx - 1);
    //
    //                if (next_frame_read)
    //                {
    //                    if ((ret = compute_motion(blur_buf, next_blur_buf, w, h, stride, stride, &score2)))
    //                    {
    //                        sprintf(errmsg, "compute_motion (next) failed.\n");
    //                        goto fail_or_end;
    //                    }
    //                    score2 = MIN(score, score2);
    //                }
    //                else
    //                {
    //                    score2 = score;
    //                }
    //            }
}

fn compute_vif() {
    // if ((ret = compute_vif(ref_buf, dis_buf, w, h, stride, stride, &score, &score_num, &score_den, scores)))
    //            {
    //                sprintf(errmsg, "compute_vif failed.\n");
    //                goto fail_or_end;
    //            }
}
