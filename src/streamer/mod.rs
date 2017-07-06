#[cfg(feature = "camera_support")]
pub mod webcam_stream;

pub mod dummy_streamer;

pub trait Stream {
    // add code here
    fn setup() -> Self;
    fn fetch_images(&self) -> Box<Iterator<Item = Vec<u8>>>;
    fn get_resolution(&self) -> (u32, u32);
}
