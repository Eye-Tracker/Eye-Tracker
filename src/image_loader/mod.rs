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

/// Column mayor (Column, row)
pub fn to_vector(img: DynamicImage) -> Vec<u8> {
    let w = img.dimensions().0;
    let h = img.dimensions().1;
    let mut res = vec![0u8; (w * h) as usize];

    for (x, y, pixel) in img.pixels() {
        res[(x * h + y) as usize] = pixel.data[0];
    }

    res
}

pub fn crop_image(img: &mut DynamicImage, desired_dim: (u32, u32)) -> DynamicImage {
    img.crop(0, 0, desired_dim.0, desired_dim.1)
}