pub extern crate image;

use std::path::Path;

use self::image::{GenericImage, DynamicImage};

pub fn open_image(file: &str) -> DynamicImage {
    let img = image::open(&Path::new(file)).unwrap();
    let grayscaled = img.grayscale();

    println!("dimensions {:?}", grayscaled.dimensions());
    println!("{:?}", grayscaled.color());

    grayscaled
}

pub fn save_image(data: Vec<u8>, dimx: u32, dimy: u32) {
    image::save_buffer(&Path::new("output.jpg"), &data, dimx, dimy, image::Gray(8)).unwrap();
}