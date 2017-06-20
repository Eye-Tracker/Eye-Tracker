pub extern crate image;

use std::path::Path;

use self::image::DynamicImage;

pub fn open_image(file: &str) -> DynamicImage {
    let img = image::open(&Path::new(file)).unwrap();
    let grayscaled = img.grayscale();

    grayscaled
}

pub fn save_image(data: Vec<u8>, dimx: u32, dimy: u32) {
    image::save_buffer(&Path::new("output.jpg"), &data, dimx, dimy, image::Gray(8)).unwrap();
}

pub fn crop_image(img: &mut DynamicImage, desired_dim: (u32, u32)) -> DynamicImage {
    img.crop(0, 0, desired_dim.0, desired_dim.1)
}