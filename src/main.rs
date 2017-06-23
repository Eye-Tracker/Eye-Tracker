extern crate image;
extern crate ocl;
extern crate find_folder;
extern crate time;
extern crate camera_capture;
extern crate texture;
extern crate piston_window;

mod image_loader;
mod opencl;
mod streamer;

use image::{GenericImage, DynamicImage, ImageBuffer};
use streamer::webcam_stream::webcam_steam;
use streamer::Stream;
use piston_window::{PistonWindow, Texture, WindowSettings, TextureSettings, clear};

fn main() {
    //let mut img = image_loader::open_image("eye.jpg");

    let window: PistonWindow =
        WindowSettings::new("Eye Detection", [640, 480])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut tex: Option<Texture<_>> = None;

    let streamer: webcam_steam = streamer::Stream::setup();
    let dim = streamer.get_resolution();
    let stream_receiver = streamer.fetch_images();

    for e in window {
        if let Ok(frame) = stream_receiver.try_recv() {
            println!("Received frame");
            let imgBuf = ImageBuffer::from_raw(dim.0, dim.1, frame).unwrap();
            let imgBufGray = DynamicImage::ImageLuma8(imgBuf).to_rgba();
            if let Some(mut t) = tex {
                t.update(&mut *e.encoder.borrow_mut(), &imgBufGray).unwrap();
                tex = Some(t);
            } else {
                tex = Texture::from_image(&mut *e.factory.borrow_mut(), &imgBufGray, &TextureSettings::new()).ok();
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

    //let mut processor = opencl::imgproc::new(true, img.dimensions());

    //loop {
    //    let imgRaw = stream_receiver.recv();
        //let img = image::load_from_memory(&imgRaw);
    //    println!("Received img");
    //};

    //let res_img = processor.execute_edge_detection(img.raw_pixels());

    //image_loader::save_image(res_img,  img.dimensions().0, img.dimensions().1);
}
