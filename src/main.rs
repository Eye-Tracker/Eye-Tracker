mod image_loader;
mod opencl;

use image_loader::image::GenericImage;

fn main() {
    let img = image_loader::open_image("eye.jpg");

    //opencl::info::print_info();

    let img_proc = opencl::imgproc::ImgPrc::new(false).process_image(img.raw_pixels(), img.dimensions());
}
