mod image_loader;
mod opencl;

fn main() {
    let img = image_loader::open_image("eye.jpg");

    //opencl::info::print_info();

    let img_proc = opencl::imgproc::ImgPrc::new(false);
}
