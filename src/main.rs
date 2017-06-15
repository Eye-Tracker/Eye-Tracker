mod image_loader;
mod canny;

fn main() {
    let img = image_loader::open_image("eye.jpg");

    canny::info::print_info();
}
