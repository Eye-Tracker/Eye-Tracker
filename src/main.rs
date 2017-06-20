mod image_loader;
mod opencl;

use image_loader::image::GenericImage;

fn main() {
    // image are row major
    let mut img = image_loader::open_image("eye.jpg");

    let mut processor = opencl::imgproc::new(true, img.dimensions());

    let new_dim = processor.get_desired_size();

    let cropped = image_loader::crop_image(&mut img, processor.get_desired_size());

    //image_loader::save_image(cropped.raw_pixels(), new_dim.0, new_dim.1);
    let res_img = processor.execute_edge_detection(cropped.raw_pixels());

    image_loader::save_image(res_img,  new_dim.0, new_dim.1);
}
