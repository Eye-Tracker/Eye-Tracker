use contour_detection::Coordinates;
use std;
use std::cell::RefCell;

pub trait Shape {
    fn calculate_area(&self) -> f64;
    fn as_polygon(&self) -> Polygon;
}

pub struct PointList {
    pub points: RefCell<Vec<Coordinates>>
}

impl Clone for PointList {
    fn clone(&self) -> PointList {
        let points = self.points.borrow().to_vec();
        PointList { points: RefCell::new(points) }
    }
}

pub trait Points {
    fn calculate_bounding_box(&self) -> Rectangle;
    fn get_vertices(&self) -> Vec<Coordinates>;
    fn min_x(&self) -> Option<usize>;
    fn min_y(&self) -> Option<usize>;
    fn max_x(&self) -> Option<usize>;
    fn max_y(&self) -> Option<usize>;
    fn width(&self) -> Option<usize>;
    fn height(&self) -> Option<usize>;
    fn add_points(&mut self, Vec<Coordinates>);
}

impl PointList {
    pub fn new(points: Vec<Coordinates>) -> PointList {
        PointList{ points: RefCell::new(points) }
    }
}

impl Points for PointList {
    fn calculate_bounding_box(&self) -> Rectangle {
        Rectangle::new(self.min_x().unwrap(), self.min_y().unwrap(), self.width().unwrap(), self.height().unwrap())
    }

    fn min_x(&self) -> Option<usize> {
        self.points.borrow().iter()
            .map(|&p| p.x)
            .min()
    }

    fn min_y(&self) -> Option<usize> {
        self.points.borrow().iter()
            .map(|&p| p.y)
            .min()
    }
    fn max_x(&self) -> Option<usize> {
        self.points.borrow().iter()
            .map(|&p| p.x)
            .max()
    }
    fn max_y(&self) -> Option<usize> {
        self.points.borrow().iter()
            .map(|&p| p.y)
            .max()
    }
    fn width(&self) -> Option<usize> {
        if let Some(max) = self.max_x() {
            if let Some(min) = self.min_x() {
                return Some(max - min);
            }
        }
        return None;
    }

    fn height(&self) -> Option<usize> {
        if let Some(max) = self.max_y() {
            if let Some(min) = self.min_y() {
                return Some(max - min);
            }
        }
        return None;
    }

    fn get_vertices(&self) -> Vec<Coordinates> {
        self.points.borrow().to_vec()
    }

    fn add_points(&mut self, list: Vec<Coordinates>) {
        self.points.borrow_mut().extend(list.iter().cloned());
    }
}

#[derive(Clone)]
pub struct Polygon {
    pointlist: PointList,
}

impl Shape for Polygon {
    fn as_polygon(&self) -> Polygon {
        self.clone()
    }

    fn calculate_area(&self) -> f64 {
        self.calculate_signed_area().abs()
    }
}

impl std::ops::Deref for Polygon {
    type Target = PointList;
    fn deref(&self) -> &Self::Target {
        &self.pointlist
    }
}

impl Polygon {
    pub fn new(list: Vec<Coordinates>) -> Self {
        Polygon{ pointlist: PointList::new(list) }
    }

    fn calculate_signed_area(&self) -> f64 {
        let closed = self.is_closed();
        let mut area = 0f64;

        let mut clone = self.clone();
        if !closed {
            clone.close();
        }

        for k in 0..(clone.points.borrow().len() - 1) {
            let ik = clone.points.borrow()[k].x;
            let jk = clone.points.borrow()[k].y;
            let ik1 = clone.points.borrow()[k+1].x;
            let jk1 = clone.points.borrow()[k+1].y;

            area = area + (ik * jk1 - ik1 * jk) as f64;
        }

        return 0.5 * area;
    }

    fn is_closed(&self) -> bool {
        self.points.borrow().len() > 0 && self.points.borrow()[0].x == self.points.borrow().last().unwrap().x
            && self.points.borrow()[0].y == self.points.borrow().last().unwrap().y
    }

    fn close(&mut self) {
        if !self.is_closed() && self.points.borrow().len() > 0 {
            self.points.borrow_mut().push(self.points.borrow()[0]);
        }
    }
}


#[derive(Clone)]
pub struct Rectangle {
    pointlist: PointList,
    /// Top left x coordinate
    x: usize,
    /// Top left y coordinate
    y: usize,
    width: usize,
    height: usize,
}

fn create_points(x: usize, y: usize, w: usize, h: usize) -> Vec<Coordinates> {
    let p1 = Coordinates::new(x,y);
    let p2 = Coordinates::new(x+w,y);
    let p3 = Coordinates::new(x+w,y+h);
    let p4 = Coordinates::new(x, y+h);
    vec![p1, p2, p3, p4]
}
impl Rectangle {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Rectangle {
        let pl = PointList::new(create_points(x, y, width, height));
        Rectangle{ pointlist: pl, x: x, y: y, width: width, height: height }
    }
}

impl std::ops::Deref for Rectangle {
    type Target = PointList;
    fn deref(&self) -> &Self::Target {
        &self.pointlist
    }
}

impl Shape for Rectangle {
    fn calculate_area(&self) -> f64 {
        (self.width * self.height) as f64
    }
    fn as_polygon(&self) -> Polygon {
        Polygon::new(self.pointlist.points.borrow().to_vec())
    }
}