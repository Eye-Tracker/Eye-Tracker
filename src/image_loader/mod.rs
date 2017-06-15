extern crate image;

use std::path::Path;

use self::image::{GenericImage, DynamicImage};

pub fn open_image(file: &str) -> DynamicImage {
    let img = image::open(&Path::new(file)).unwrap();
    let grayscaled = img.grayscale();

    println!("dimensions {:?}", grayscaled.dimensions());
    println!("{:?}", grayscaled.color());

    grayscaled
}