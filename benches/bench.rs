extern crate av_metrics;
#[macro_use]
extern crate criterion;

use av_metrics::video;
use av_metrics::video::decode::Decoder;
use av_metrics::video::pixel::Pixel;
use av_metrics::video::FrameInfo;
use criterion::Criterion;
use std::fs::File;
use y4m::Decoder as Y4MDec;

fn get_video_frame<T: Pixel>(filename: &str) -> FrameInfo<T> {
    let mut file = File::open(filename).unwrap();
    let mut dec = Y4MDec::new(&mut file).unwrap();
    dec.read_video_frame().unwrap()
}

pub fn psnr_benchmark(c: &mut Criterion) {
    let frame1 = get_video_frame::<u8>("./testfiles/yuv420p8_input.y4m");
    let frame2 = get_video_frame::<u8>("./testfiles/yuv420p8_output.y4m");
    c.bench_function("PSNR yuv420p8", |b| {
        b.iter(|| {
            video::calculate_frame_psnr(&frame1, &frame2).unwrap();
        })
    });
}

pub fn psnrhvs_benchmark(c: &mut Criterion) {
    let frame1 = get_video_frame::<u8>("./testfiles/yuv420p8_input.y4m");
    let frame2 = get_video_frame::<u8>("./testfiles/yuv420p8_output.y4m");
    c.bench_function("PSNR-HVS yuv420p8", |b| {
        b.iter(|| {
            video::calculate_frame_psnr_hvs(&frame1, &frame2).unwrap();
        })
    });
}

pub fn ssim_benchmark(c: &mut Criterion) {
    let frame1 = get_video_frame::<u8>("./testfiles/yuv420p8_input.y4m");
    let frame2 = get_video_frame::<u8>("./testfiles/yuv420p8_output.y4m");
    c.bench_function("SSIM", |b| {
        b.iter(|| {
            video::calculate_frame_ssim(&frame1, &frame2).unwrap();
        })
    });
}

pub fn msssim_benchmark(c: &mut Criterion) {
    let frame1 = get_video_frame::<u8>("./testfiles/yuv420p8_input.y4m");
    let frame2 = get_video_frame::<u8>("./testfiles/yuv420p8_output.y4m");
    c.bench_function("MSSSIM", |b| {
        b.iter(|| {
            video::calculate_frame_msssim(&frame1, &frame2).unwrap();
        })
    });
}

pub fn ciede2000_nosimd_benchmark(c: &mut Criterion) {
    let frame1 = get_video_frame::<u8>("./testfiles/yuv420p8_input.y4m");
    let frame2 = get_video_frame::<u8>("./testfiles/yuv420p8_output.y4m");
    c.bench_function("CIEDE2000", |b| {
        b.iter(|| {
            video::calculate_frame_ciede(&frame1, &frame2, false).unwrap();
        })
    });
}

pub fn ciede2000_simd_benchmark(c: &mut Criterion) {
    let frame1 = get_video_frame::<u8>("./testfiles/yuv420p8_input.y4m");
    let frame2 = get_video_frame::<u8>("./testfiles/yuv420p8_output.y4m");
    c.bench_function("CIEDE2000", |b| {
        b.iter(|| {
            video::calculate_frame_ciede(&frame1, &frame2, true).unwrap();
        })
    });
}

criterion_group!(
    benches,
    psnr_benchmark,
    psnrhvs_benchmark,
    ssim_benchmark,
    msssim_benchmark,
    ciede2000_nosimd_benchmark,
    ciede2000_simd_benchmark
);
criterion_main!(benches);
