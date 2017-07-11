use streamer::Stream;
use image_loader;
use std::iter;
use image;

#[derive(Clone)]
pub struct DummyStream{
    dim: (u32, u32),
    img: image::RgbImage,
}

impl Stream for DummyStream {
    fn setup() -> DummyStream {
        let img = image_loader::open_image("eye.jpg").to_rgb();
        DummyStream{ dim: (img.width(), img.height()), img: img }
    }

    fn fetch_images(&self) -> Box<Iterator<Item = image::RgbImage>> {
        let img = self.img.clone();
        Box::new(iter::repeat(img))
    }

    fn get_resolution(&self) -> (u32, u32) {
        self.dim
    }
}