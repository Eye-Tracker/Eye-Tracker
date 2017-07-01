use std::sync::mpsc::{channel, Receiver};
use std::thread;
use streamer::Stream;
use camera_capture;

#[derive(Clone)]
pub struct WebcamStream{
    dim: (u32, u32)
}

impl Stream for WebcamStream {
    fn setup() -> WebcamStream {
        WebcamStream{ dim: (640, 480) }
    }

    fn fetch_images<'a>(&self) -> (thread::JoinHandle<()>, Receiver<Vec<u8>>) {
        let (sender, receiver) = channel();
        let self_clone = self.clone();

        let handler = thread::spawn(move || {
            let cam = camera_capture::create(0).unwrap()
                                        .fps(20.0)
                                        .unwrap()
                                        .resolution(self_clone.dim.0, self_clone.dim.1)
                                        .unwrap()
                                        .start()
                                        .unwrap();

            for frame in cam {
                if let Err(_) = sender.send(frame.into_raw()) {
                    println!("Image sending failed!");
                    break;
                }
            };
        });

        (handler, receiver)
    }

    fn get_resolution(&self) -> (u32, u32) {
        self.dim
    }
}