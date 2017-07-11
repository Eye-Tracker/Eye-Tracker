pub mod direction;
pub mod shape;
pub mod contour;
pub mod contour_processor;

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