#[macro_use()]
macro_rules! calculate_block_size {
    ($a:expr, $b:expr) => (($a/$b+1)*$b);
}

pub mod imgproc;
mod canny_edge_detection;
mod ellipse_detect;