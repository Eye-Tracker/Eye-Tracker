pub mod direction;
pub mod shape;
pub mod contour;
pub mod contour_processor;
pub mod contour_follower;

use image::GrayImage;
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub enum ContourType {
    Hole, //Hole with an enclosing contour
    Outer, //Outer contour
}

#[derive(Clone, Copy, Debug)]
pub struct Coordinates {
    pub x: usize,
    pub y: usize,
}

impl Coordinates {
    pub fn new(x: usize, y: usize) -> Coordinates {
        Coordinates{ x: x, y: y }
    }

    pub fn eq_y(&self, other: &Coordinates) -> bool {
        self.y == other.y
    }

    pub fn eq_x(&self, other: &Coordinates) -> bool {
        self.x == other.x
    }

    pub fn lt_y(&self, other: &Coordinates) -> bool {
        self.y < other.y
    }

    pub fn lt_x(&self, other: &Coordinates) -> bool {
        self.x < other.x
    }

    pub fn gt_y(&self, other: &Coordinates) -> bool {
        self.y > other.y
    }

    pub fn gt_x(&self, other: &Coordinates) -> bool {
        self.x > other.x
    }
}

impl PartialEq for Coordinates {
    fn eq(&self, other: &Coordinates) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Coordinates {}

pub struct FloatImage {
    data: RefCell<Vec<f32>>,
    width: u32,
    height: u32,
}

impl FloatImage {
    pub fn new(img: &GrayImage) -> FloatImage {
        let height = img.height();
        let width = img.width();
        let mut data = vec![0f32; (width * height) as usize];

        for y in 0..img.height() {
            for x in 0..img.width() {
                data[(y * width + x) as usize] = img.get_pixel(x, y).data[0] as f32 / 255f32;
            }
        }

        FloatImage{data: RefCell::new(data), width: width, height: height }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> f32 {
        self.data.borrow()[(y * self.width + x) as usize]
    }

    pub fn set_pixel(&self, x: u32, y: u32, value: f32) {
        self.data.borrow_mut()[(y * self.width + x) as usize] = value;
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}