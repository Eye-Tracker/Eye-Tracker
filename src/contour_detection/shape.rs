use contour_detection::Coordinates;
use std;

pub trait Shape {
    fn is_inside(&self, p: Coordinates) -> bool;
    //fn calculate_area(&self) -> f64;
    //fn calculate_perimeter(&self) -> f64;
    fn as_polygon(&self) -> Polygon;
    //fn intersection_area(&self, shape: Shape) -> f64;
    //fn is_convex(&self);
}

pub struct PointList {
    pub points: Vec<Coordinates>
}

impl Clone for PointList {
    fn clone(&self) -> PointList {
        let points = self.points.to_vec();
        PointList { points: points }
    }
}

pub trait Points {
    //fn translate(&self) -> Self;
    //fn rotate(&self, point: Coordinates, origin: Coordinates, angle: f64) -> Self;
    //fn scale(&self) -> Self;
    //fn scale_x(&self) -> Self;
    //fn scale_y(&self) -> Self;
    fn calculate_bounding_box(&self) -> Rectangle;
    //fn calculate_centroid(&self) -> Coordinates;
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
        PointList{ points: points }
    }
}

impl Points for PointList {
    fn calculate_bounding_box(&self) -> Rectangle {
        Rectangle::new(self.min_x().unwrap(), self.min_y().unwrap(), self.width().unwrap(), self.height().unwrap())
    }

    fn min_x(&self) -> Option<usize> {
        self.points.iter()
            .map(|&p| p.x)
            .min()
    }

    fn min_y(&self) -> Option<usize> {
        self.points.iter()
            .map(|&p| p.y)
            .min()
    }
    fn max_x(&self) -> Option<usize> {
        self.points.iter()
            .map(|&p| p.x)
            .max()
    }
    fn max_y(&self) -> Option<usize> {
        self.points.iter()
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
        self.points.to_vec()
    }

    fn add_points(&mut self, list: Vec<Coordinates>) {
        self.points.extend(list.iter().cloned());
    }
}

#[derive(Clone)]
pub struct Polygon {
    pointlist: PointList,
}

impl Shape for Polygon {
    fn as_polygon(&self) -> Polygon {
        self.clone() //TODO implement clone
    }

    fn is_inside(&self, p: Coordinates) -> bool {
        false
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

    pub fn get_vertices(&self) ->  Vec<Coordinates> {
        self.pointlist.points.to_vec()
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

   // pub fn new(coords: Coordinates, width: usize, height: usize) {
   //     Rectangle::new(coords.x, coords.y, width, height)
    //}

    // Create a Rectangle from top right (x,y) coordinates and bottom right (x,y) coordinates
    //pub fn new(topLeft: Coordinates, bottomRight: Coordinates) -> Self {
   //     Rectangle::new(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y)
   // }

     
}

impl std::ops::Deref for Rectangle {
    type Target = PointList;
    fn deref(&self) -> &Self::Target {
        &self.pointlist
    }
}

impl Shape for Rectangle {
    fn is_inside(&self, p: Coordinates) -> bool {
        p.x >= self.x && p.x <= self.x + self.width && 
            p.y >= self.y && p.y <= self.y + self.height
    }
    //fn calculate_area(&self) -> f64;
    //fn calculate_perimeter(&self) -> f64;
    fn as_polygon(&self) -> Polygon {
        Polygon::new(self.pointlist.points.to_vec())
    }
    //pub fn intersection_area(&self, shape: Shape) -> f64;
    //pub fn is_convex(&self);
}