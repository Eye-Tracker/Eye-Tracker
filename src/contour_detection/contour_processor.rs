use contour_detection::contour::{Contour, ContourBuilder};
use contour_detection::ContourType;
use contour_detection::shape::{Rectangle, Shape, Polygon, Points};
use image::GrayImage;
use indextree::Arena;

pub struct ContourProcessor{
    root: Contour,
}

impl ContourProcessor {
    pub fn findContours(&mut self, img: GrayImage) {
        let bounds = Rectangle::new(0, 0, img.width() as usize, img.height() as usize);
        
        let arena = &mut Arena::<Contour>::new();

        let root = arena.new_node(
            ContourBuilder::new()
                .set_type(ContourType::Hole)
                .set_points(bounds.as_polygon().get_vertices()).finish());
        
        

        


    }
}