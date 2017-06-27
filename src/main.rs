extern crate image;
extern crate ocl;
extern crate find_folder;
extern crate time;
extern crate camera_capture;
extern crate texture;
extern crate piston_window;
extern crate fps_counter;

mod image_loader;
mod opencl;
mod streamer;

use image::{DynamicImage, ImageBuffer};
use streamer::webcam_stream::WebcamStream;
use streamer::Stream;
use piston_window::{PistonWindow, Texture, WindowSettings, TextureSettings, clear};
use fps_counter::FPSCounter;

fn main() {
    let window: PistonWindow =
        WindowSettings::new("Eye Detection", [640, 480])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut tex: Option<Texture<_>> = None;

    let streamer: WebcamStream = streamer::Stream::setup();
    let dim = streamer.get_resolution();
    let (stream_handler, stream_receiver) = streamer.fetch_images();

    let mut processor = opencl::imgproc::new(true, dim);

    let mut fpsc = FPSCounter::new();

    for e in window {
        if let Ok(frame) = stream_receiver.try_recv() {
            println!("FPS: {}", fpsc.tick());
            let img_buf: image::GrayImage = ImageBuffer::from_raw(dim.0, dim.1, frame).unwrap();
            let res_raw = processor.execute_edge_detection(img_buf.into_vec());

            let res_buf: image::GrayImage = ImageBuffer::from_raw(dim.0, dim.1, res_raw).unwrap();
            let res_img: image::RgbaImage = DynamicImage::ImageLuma8(res_buf).to_rgba();

            if let Some(mut t) = tex {
                t.update(&mut *e.encoder.borrow_mut(), &res_img).unwrap();
                tex = Some(t);
            } else {
                tex = Texture::from_image(&mut *e.factory.borrow_mut(), &res_img, &TextureSettings::new()).ok();
            }
        }
        e.draw_2d(|c, g| {
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
