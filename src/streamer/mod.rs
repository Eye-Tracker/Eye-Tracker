pub mod webcam_stream;

use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

pub trait Stream {
    // add code here
    fn setup() -> Self;
    fn fetch_images(&self) -> (JoinHandle<()>, Receiver<Vec<u8>>);
    fn get_resolution(&self) -> (u32, u32);
}
