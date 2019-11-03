use crate::video::pixel::Pixel;
use crate::video::{CastFromPrimitive, ChromaSampling, FrameInfo};
use crate::MetricsError;
use libc::*;
use std::error::Error;

pub(crate) fn compute_vmaf<T: Pixel>(
    frame1: &FrameInfo<T>,
    frame2: &FrameInfo<T>,
    use_simd: bool,
) -> Result<f64, Box<dyn Error>> {
    let mut score = 0f64;
    let format = match (frame1.bit_depth, frame1.chroma_sampling) {
        (8, ChromaSampling::Cs420) => "yuv420p",
        (8, ChromaSampling::Cs422) => "yuv422p",
        (8, ChromaSampling::Cs444) => "yuv444p",
        (10, ChromaSampling::Cs420) => "yuv420p10le",
        (10, ChromaSampling::Cs422) => "yuv422p10le",
        (10, ChromaSampling::Cs444) => "yuv444p10le",
        _ => {
            return Err(Box::new(MetricsError::UnsupportedInput {
                reason: "VMAF only supports 8-bit and 10-bit inputs",
            }))
        }
    };
    unsafe {
        let width = frame1.planes[0].width;
        let height = frame1.planes[0].height;
        let model_path = if width >= 2880 {
            // This is somewhat of an arbitrary cutoff,
            // but it attempts to handle videos between 1080p and 4K
            // in a reasonable manner.
            vmaf_sys::extras::get_4k_model_path()
        } else {
            vmaf_sys::extras::get_def_model_path()
        }
        .to_str()
        .unwrap()
        .to_owned();
        let mut user_data = Data {
            format: format.as_ptr() as *mut c_char,
            width: width as c_int,
            height: height as c_int,
            offset: match frame1.chroma_sampling {
                ChromaSampling::Cs420 => width * height / 2,
                ChromaSampling::Cs422 => width * height,
                ChromaSampling::Cs444 => width * height * 2,
                _ => unreachable!(),
            } as size_t,
            ref_rfile: None,
            dis_rfile: None,
            num_frames: 1,
        };

        let read_frame = |ref_data: *mut f32,
                          main_data: *mut f32,
                          temp_data: *mut f32,
                          stride_byte: c_int,
                          user_data: *mut c_void|
         -> c_int {
            for (i, (&pix1, &pix2)) in frame1.planes[0]
                .data
                .iter()
                .zip(frame2.planes[0].data.iter())
                .enumerate()
            {
                // Ferris disapproves ಠ_ಠ
                *ref_data.add(i) = u16::cast_from(pix1) as f32;
                *main_data.add(i) = u16::cast_from(pix2) as f32;
            }
            0
        };

        let result = vmaf_sys::compute_vmaf(
            &mut score,
            format.as_ptr() as *mut c_char,
            width as c_int,
            height as c_int,
            Some(read_frame),
            // Netflix developers using void* in 2019 ಠ_ಠ
            (&mut user_data as *mut Data).cast::<c_void>(),
            model_path.as_ptr() as *mut c_char,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            (!use_simd) as c_int,
            0,
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            0,
            1,
            0,
        );
    }
    Ok(score)
}

#[repr(C)]
struct Data {
    pub format: *mut c_char,
    pub width: c_int,
    pub height: c_int,
    pub offset: size_t,
    pub ref_rfile: Option<*mut FILE>,
    pub dis_rfile: Option<*mut FILE>,
    pub num_frames: c_int,
}
