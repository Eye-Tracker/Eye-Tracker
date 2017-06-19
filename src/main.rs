mod image_loader;
mod opencl;

use image_loader::image::GenericImage;

fn main() {
    let mut img = image_loader::open_image("eye.jpg");

    let mut processor = opencl::imgproc::new(true, img.dimensions());

    let old_dim = img.dimensions();

    let cropped = image_loader::crop_image(&mut img, processor.get_desired_size());
    let conv = image_loader::to_vector(cropped);

    let res_img = processor.execute_edge_detection(conv);

    image_loader::save_image(res_img, img.width(), img.height());
}
