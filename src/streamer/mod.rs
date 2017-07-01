#[cfg(feature = "camera_support")]
pub mod webcam_stream;

pub mod dummy_streamer;

use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

pub trait Stream {
    // add code here
    fn setup() -> Self;
    fn fetch_images(&self) -> (JoinHandle<()>, Receiver<Vec<u8>>);
    fn get_resolution(&self) -> (u32, u32);
}
