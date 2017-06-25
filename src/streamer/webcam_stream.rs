use std::sync::mpsc::{channel, Receiver};
use std::thread;
use streamer::Stream;
use camera_capture;
use image::{DynamicImage, Rgb, Pixel};
use image::imageops::colorops;

#[derive(Clone)]
pub struct webcam_steam{
    dim: (u32, u32)
}

impl Stream for webcam_steam {
    fn setup() -> webcam_steam {
        webcam_steam{ dim: (640, 480) }
    }

    fn fetch_images(&self) -> (thread::JoinHandle<()>, Receiver<Vec<u8>>) {
        let (sender, receiver) = channel();
        let self_clone = self.clone();

        let handler = thread::spawn(move || {
            let cam = camera_capture::create(0).unwrap()
                                        .fps(30.0)
                                        .unwrap()
                                        .resolution(self_clone.dim.0, self_clone.dim.1)
                                        .unwrap()
                                        .start()
                                        .unwrap();

            for frame in cam {
                let gray = DynamicImage::ImageRgb8(frame).grayscale();
                if let Err(_) = sender.send(gray.raw_pixels()) {
                    println!("Image sending failed!");
                    break;
                }
            }
            println!("No more images in queue");
        });

        println!("End of fetch images");

        (handler, receiver)
    }

    fn get_resolution(&self) -> (u32, u32) {
        self.dim
    }
}