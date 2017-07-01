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

use image::ImageBuffer;
use fps_counter::FPSCounter;

#[cfg(feature = "camera_support")]
use streamer::webcam_stream::WebcamStream;

use streamer::dummy_streamer::DummyStream;
use streamer::Stream;
use piston_window::*;
use std::thread::JoinHandle;
use std::sync::mpsc::Receiver;

#[cfg(feature = "camera_support")]
fn setup_streamer() -> (JoinHandle<()>, Receiver<Vec<u8>>, (u32, u32)) {
        let streamer: WebcamStream = streamer::Stream::setup();
        let dim = streamer.get_resolution();
        let (stream_handler, stream_receiver) = streamer.fetch_images();
        (stream_handler, stream_receiver, dim)
}

#[cfg(feature = "use_dummy_streamer")]
fn setup_streamer() -> (JoinHandle<()>, Receiver<Vec<u8>>, (u32, u32)) {
    let streamer: DummyStream = streamer::Stream::setup();
    let dim = streamer.get_resolution();
    let (stream_handler, stream_receiver) = streamer.fetch_images();
    (stream_handler, stream_receiver, dim)
}

fn main() {
    let (stream_handler, stream_receiver, dim) = setup_streamer();
    let mut processor = opencl::imgproc::new(true, dim);

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("Eye Detection", [dim.0, dim.1])
        .exit_on_esc(true)
        .opengl(opengl)
        .build()
        .unwrap();

    let assets = find_folder::Search::ParentsThenKids(3, 3)
        .for_folder("assets").unwrap();
    let ref font = assets.join("FiraSans-Regular.ttf");
    let factory = window.factory.clone();
    let mut glyphs = Glyphs::new(font, factory).unwrap();

    let mut tex: Option<Texture<_>> = None;
    let mut fpsc = FPSCounter::new();

    let limbus_pos = [
            dim.0 as f64 / 2.5,
            dim.1 as f64 / 3.5,
            dim.0 as f64 / 3.0,
            dim.0 as f64 / 3.0];

    while let Some(e) = window.next() {
        if let Ok(frame) = stream_receiver.try_recv() {
            let res_raw = processor.execute_edge_detection(frame);

            let res_img: image::RgbaImage = ImageBuffer::from_raw(dim.0, dim.1, res_raw).unwrap();

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
                Ellipse::new_border([1.0, 0.0, 0.0, 1.0], 1f64)
                    .draw(limbus_pos, &c.draw_state, c.transform, g);
                let transform = c.transform.trans(10.0, 30.0);
                text::Text::new_color([0.29, 0.68, 0.31, 1.0], 24).draw(
                    &format!("FPS: {}", fpsc.tick()),
                    &mut glyphs,
                    &c.draw_state,
                    transform, g
                );
            }
        });
    }

    println!("Done");
    drop(stream_receiver);
    stream_handler.join().unwrap();
}
