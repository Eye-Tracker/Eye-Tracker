use streamer::Stream;
use image::{GenericImage, DynamicImage};
use image_loader;
use std::iter;

#[derive(Clone)]
pub struct DummyStream{
    dim: (u32, u32),
    img: DynamicImage
}

impl Stream for DummyStream {
    fn setup() -> DummyStream {
        let img = image_loader::open_image("eye.jpg");
        DummyStream{ dim: (img.width(), img.height()), img: img }
    }

    fn fetch_images(&self) -> Box<Iterator<Item = Vec<u8>>> {
        Box::new(iter::repeat(self.img.raw_pixels()))
    }

    fn get_resolution(&self) -> (u32, u32) {
        self.dim
    }
}