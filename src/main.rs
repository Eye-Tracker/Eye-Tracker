mod image_loader;
mod opencl;

use image_loader::image::GenericImage;

fn main() {
    let img = image_loader::open_image("eye.jpg");

    let res_img = opencl::imgproc::ImgPrc::new(true)
        .process_image(img.raw_pixels(), img.dimensions());

    image_loader::save_image(res_img, img.width(), img.height());
}
