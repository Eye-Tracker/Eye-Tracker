extern crate image;
extern crate ocl;
extern crate find_folder;
extern crate time;
extern crate texture;
extern crate piston_window;
extern crate fps_counter;

#[cfg(feature = "camera_support")]
extern crate camera_capture;

mod image_loader;
mod opencl;
mod streamer;

use image::{DynamicImage, ImageBuffer};
use fps_counter::FPSCounter;

#[cfg(feature = "camera_support")]
use streamer::webcam_stream::webcam_steam;

use streamer::dummy_streamer::dummy_stream;
use streamer::Stream;
use piston_window::*;
use std::thread::JoinHandle;
use std::sync::mpsc::Receiver;

#[cfg(feature = "camera_support")]
fn setup_streamer() -> (JoinHandle<()>, Receiver<Vec<u8>>, (u32, u32)) {
        let streamer: webcam_steam = streamer::Stream::setup();
        let dim = streamer.get_resolution();
        let (stream_handler, stream_receiver) = streamer.fetch_images();
        (stream_handler, stream_receiver, dim)
}

#[cfg(feature = "use_dummy_streamer")]
fn setup_streamer() -> (JoinHandle<()>, Receiver<Vec<u8>>, (u32, u32)) {
    let streamer: dummy_stream = streamer::Stream::setup();
    let dim = streamer.get_resolution();
    let (stream_handler, stream_receiver) = streamer.fetch_images();
    (stream_handler, stream_receiver, dim)
}


fn main() {
    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("Eye Detection", [640, 480])
        .exit_on_esc(true)
        .opengl(opengl)
        .build()
        .unwrap();

    let mut tex: Option<Texture<_>> = None;
    let mut fpsc = FPSCounter::new();

    let (stream_handler, stream_receiver, dim) = setup_streamer();
    let mut processor = opencl::imgproc::new(true, dim);

    while let Some(e) = window.next() {
        if let Ok(frame) = stream_receiver.try_recv() {
            println!("FPS: {}", fpsc.tick());
            let img_buf: image::GrayImage = ImageBuffer::from_raw(dim.0, dim.1, frame).unwrap();
            let res_raw = processor.execute_edge_detection(img_buf.into_vec());

            let res_buf: image::GrayImage = ImageBuffer::from_raw(dim.0, dim.1, res_raw).unwrap();
            let res_img: image::RgbaImage = DynamicImage::ImageLuma8(res_buf).to_rgba();

            if let Some(mut t) = tex {
                t.update(&mut window.encoder, &res_img).unwrap();
                tex = Some(t);
            } else {
                tex = Texture::from_image(&mut window.factory, &res_img, &TextureSettings::new()).ok();
            }
        }
        window.draw_2d(&e, |c, g| {
            clear([1.0; 4], g);
            if let Some(ref t) = tex {
                piston_window::image(t, c.transform, g);
            }
        });
    }

    println!("Done");
    drop(stream_receiver);
    stream_handler.join().unwrap();
}
