use contour_detection::Coordinates;
use contour_detection::ContourType;
use contour_detection::shape::{PointList, Polygon, Rectangle, Shape, Points};

pub struct Contour {
    ctype: Option<ContourType>,
    start: Coordinates,
    points: Polygon,
    bounds: Rectangle,
}

pub struct ContourBuilder {
    ctype: Option<ContourType>,
    start: Coordinates,
    points: Option<Polygon>,
}

impl ContourBuilder {
    pub fn new() -> Self {
        ContourBuilder{ ctype: None, start: Coordinates::new(0, 0), points: None }
    }

    pub fn set_type(mut self, t: ContourType) -> Self {
        self.ctype = Some(t);
        self
    }

    pub fn set_start(mut self, c: Coordinates) -> Self {
        self.start = c;
        self
    }

    pub fn set_points(mut self, list: Vec<Coordinates>) -> Self {
        self.points = Some(Polygon::new(list));
        self
    }

    pub fn finish(self) -> Contour {
        Contour{ ctype: self.ctype, start: self.start, points: self.points.clone().unwrap(),
            bounds: self.points.unwrap().calculate_bounding_box().clone() }
    }
}