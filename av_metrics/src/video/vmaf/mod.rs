//! VMAF - Video Multi-Method Assessment Fusion
//!
//! VMAF is a perceptual video quality assessment algorithm developed by Netflix.
//! At the present, this uses Netflix's C implementation,
//! available on [Github](https://github.com/Netflix/vmaf).
//!
//! In the future, this will be Rewritten In Rust.

#[cfg(feature = "decode")]
use crate::video::decode::Decoder;
use crate::video::pixel::Pixel;
use crate::video::vmaf::native::VmafMetrics;
use crate::video::{FrameInfo, VideoMetric};
use std::error::Error;

//mod ffi;
mod models;
mod native;

/// Calculate the VMAF metric between two video clips. Higher is better.
///
/// This will return at the end of the shorter of the two clips,
/// comparing any frames up to that point.
///
/// Optionally, `frame_limit` can be set to only compare the first
/// `frame_limit` frames in each video.
#[cfg(feature = "decode")]
#[inline]
pub fn calculate_video_vmaf<D: Decoder>(
    decoder1: &mut D,
    decoder2: &mut D,
    frame_limit: Option<usize>,
) -> Result<f64, Box<dyn Error>> {
    Vmaf::default().process_video(decoder1, decoder2, frame_limit)
}

/// Calculate the VMAF metric between two video frames. Higher is better.
#[inline]
pub fn calculate_frame_vmaf<T: Pixel>(
    frame1: &FrameInfo<T>,
    frame2: &FrameInfo<T>,
) -> Result<f64, Box<dyn Error>> {
    Vmaf::default().process_frame(frame1, frame2)
}

struct Vmaf {
    use_simd: bool,
    previous_frame: Option<VmafMetrics>,
}

impl Default for Vmaf {
    fn default() -> Self {
        Vmaf {
            use_simd: true,
            previous_frame: None,
        }
    }
}

impl VideoMetric for Vmaf {
    type FrameResult = f64;
    type VideoResult = f64;

    // This is NOT CURRENTLY THREAD SAFE because it depends on previous frame data.
    fn process_frame<T: Pixel>(
        &mut self,
        frame1: &FrameInfo<T>,
        frame2: &FrameInfo<T>,
    ) -> Result<Self::FrameResult, Box<dyn Error>> {
        frame1.can_compare(&frame2)?;

        let res = self::native::compute_vmaf(
            frame1,
            frame2,
            self.previous_frame.as_ref(),
            self.use_simd,
        )?;
        let score = res.score;
        self.previous_frame = Some(res);
        Ok(score)
    }

    #[cfg(feature = "decode")]
    fn aggregate_frame_results(
        &self,
        metrics: &[Self::FrameResult],
    ) -> Result<Self::VideoResult, Box<dyn Error>> {
        Ok(metrics.iter().copied().sum::<f64>() / metrics.len() as f64)
    }
}
