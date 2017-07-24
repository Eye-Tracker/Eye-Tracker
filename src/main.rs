#![feature(box_syntax, box_patterns)]

extern crate image;
extern crate ocl;
extern crate find_folder;
extern crate time;
extern crate texture;
extern crate piston;
extern crate piston_window;
extern crate fps_counter;
extern crate indextree;
#[macro_use]
extern crate conrod;
extern crate rand;
extern crate ordered_float;

#[cfg(feature = "camera_support")]
extern crate camera_capture;

mod image_loader;
mod opencl;
mod streamer;
mod contour_detection;
mod gui;

use image::ConvertBuffer;
use fps_counter::FPSCounter;
use std::thread;
use std::sync::{Arc, Mutex};

use contour_detection::contour_processor::ContourProcessor;
use contour_detection::shape::{Polygon, Points, PointList};

#[cfg(feature = "camera_support")]
use streamer::webcam_stream::WebcamStream;

#[cfg(feature = "use_dummy_streamer")]
use streamer::dummy_streamer::DummyStream;
use streamer::Stream;

use piston_window::*;
use piston::input::*;

#[cfg(feature = "camera_support")]
fn setup_streamer() -> (Box<Iterator<Item = image::RgbImage>>, (u32, u32)) {
    let streamer: WebcamStream = streamer::Stream::setup();
    let dim = streamer.get_resolution();
    let iterator = streamer.fetch_images();
    (iterator, dim)
}

#[cfg(feature = "use_dummy_streamer")]
fn setup_streamer() -> (Box<Iterator<Item = image::RgbImage>>, (u32, u32)) {
    let streamer: DummyStream = streamer::Stream::setup();
    let dim = streamer.get_resolution();
    let iterator = streamer.fetch_images();
    (iterator, dim)
}

fn main() {
    let (mut iterator, dim) = setup_streamer();
    let processor = opencl::imgproc::setup(true, dim);
    
    let mut canny_edge = processor.setup_canny_edge_detection();
    let ellipse_fit = processor.setup_ellipse_detection(2f32, 30);

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

    let app = Arc::new(Mutex::new(gui::App::new(10.0, 70.0, 0.1)));
    let data = app.clone();
    thread::spawn(move || {
        gui::draw_gui(data);
    });

    let contour_finder = ContourProcessor;
    let mut debug_view = false;
    while let Some(e) = window.next() {
        let mut fps = 0;
        let unlocked = app.lock().unwrap();
        let (low, high, size) = (unlocked.low_threshold, unlocked.high_threshold, unlocked.size_filter);
        let mut ellipses = None;

        if let Some(Button::Keyboard(key)) = e.press_args() {
            if key == Key::D {
                debug_view = !debug_view;
            }
        }

        if let Some(frame) = iterator.next() {
            fps = fpsc.tick();

            let grayscaled: image::GrayImage = frame.convert();
            let result = canny_edge.execute_edge_detection(grayscaled.into_raw(), low, high);
            let gray_result = image::GrayImage::from_raw(dim.0, dim.1, result).expect("ImageBuffer couldn't be created");
 
            let contours = contour_finder.find_contours(&gray_result, size);

            ellipses = ellipse_fit.execute_ellipse_fit(&contours);
            
            let mut rgba: image::RgbaImage = if debug_view {
                gray_result.convert()
            } else {
                frame.convert()
            };

            for c in contours { 
                for p in c.points.get_vertices() { 
                    rgba.put_pixel(p.x as u32, p.y as u32, image::Rgba{ data: [255, 0, 0, 255] }); 
                }
            }

            if let Some(mut t) = tex {
                t.update(&mut window.encoder, &rgba).unwrap();
                tex = Some(t);
            } else {
                tex = Texture::from_image(&mut window.factory, &rgba, &TextureSettings::new()).ok();
            }
        }
        window.draw_2d(&e, |c, g| {
            clear([1.0; 4], g);
            if let Some(ref t) = tex {
                piston_window::image(t, c.transform, g);
                if let Some(ref ellipses_found) = ellipses {
                    for e in ellipses_found {
                        if e.2 > 0f32 && e.0 > 0 && e.0 < dim.0 as usize && e.1 > 0 && e.1 < dim.1 as usize {
                            let pos = [ e.0 as f64 - e.2 as f64, e.1 as f64 - e.2 as f64, 
                                e.2 as f64, e.2 as f64];
                            Ellipse::new_border([1.0, 0.0, 0.0, 1.0], 1f64)
                                .draw(pos, &c.draw_state, c.transform, g);
                        }
                    }
                }
                let transform = c.transform.trans(10.0, 30.0);
                text::Text::new_color([0.29, 0.68, 0.31, 1.0], 24).draw(
                    &format!("FPS: {}", fps),
                    &mut glyphs,
                    &c.draw_state,
                    transform, g
                );
            }
        });
    }

    println!("Done");
}
