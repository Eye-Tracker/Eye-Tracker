use contour_detection::contour::{Contour, ContourBuilder};
use contour_detection::shape::{Rectangle, Shape, Polygon, Points};
use contour_detection::contour_follower::{FollowingStrategy, SuzukiStrategy};
use contour_detection::{ContourType, Coordinates};
use image::GrayImage;
use indextree::Arena;
use std::collections::HashMap;

pub struct ContourProcessor{
    remove_min_child_threshold: f64,
}

impl ContourProcessor {
    pub fn find_contours(&mut self, img: &GrayImage) {
        let bounds = Rectangle::new(0, 0, img.width() as usize, img.height() as usize);
        
        let arena = &mut Arena::<Contour>::new();

        let root = arena.new_node(
            ContourBuilder::new()
                .set_type(ContourType::Hole)
                .set_points(bounds.as_polygon().get_vertices()).finish());
        
        let border_follow = SuzukiStrategy;

        let mut nbd = 1i32;
        let mut lnbd = 1i32;

        let mut borderMap = HashMap::new();
        borderMap.insert(lnbd, root);

        for y in 0..img.height() {
            lnbd = 1i32;
            for x in 0..img.width() {
                let cur_pixel = img.get_pixel(x, y).data[0];
                let is_outer = self.is_outer_border_start(img, x as i32, y as i32);
                let is_hole = self.is_hole_border_start(img, x as i32, y as i32);
                if is_outer || is_hole {
                    
                    let mut from = Coordinates::new(x as usize, y as usize);
                    let mut border = ContourBuilder::new().set_start(from);

                    let start = Coordinates::new(x as usize, y as usize);

                    if is_outer {
                        nbd = nbd + 1;
                        from.x = from.x - 1;
                        border = border.set_type(ContourType::Outer);
                    } else {
                        nbd = nbd + 1;
                        if cur_pixel != 0 {
                            lnbd = 1;
                        }
                        from.x = from.x + 1;
                    }

                    let mut points = Vec::new();
                    border_follow.directed_contour(img, start, from, |(coord, checked)| {
                        points.push(coord);
                    });

                    if is_outer {
                        let contour = arena.new_node(border.finish());
                        if let Some(border_prime) = borderMap.get(&lnbd) {
                            match arena[border_prime.clone()].data.ctype {
                                Some(ContourType::Hole) => {
                                    border_prime.append(contour, arena)
                                },
                                Some(ContourType::Outer) => {
                                    let p_parent = border_prime.ancestors(arena).next().unwrap();
                                    p_parent.append(contour, arena);
                                },
                                None => panic!("Type should always been set!"),
                            }
                            
                        };

                        borderMap.insert(nbd, contour);
                    }
                    

                }
            }
        }
        


    }

    fn is_outer_border_start(&self, img: &GrayImage, x: i32, y: i32) -> bool {
        let color: u8 = img.get_pixel(x as u32, y as u32).data[0];
        let color1: u8 = img.get_pixel(x as u32, (y - 1) as u32).data[0];
        color != 0 && (y == 0 || color1 == 0)
    }

    fn is_hole_border_start(&self, img: &GrayImage, x: i32, y: i32) -> bool {
        let color: u8 = img.get_pixel(x as u32, y as u32).data[0];
        let color1: u8 = img.get_pixel(x as u32, (y + 1) as u32).data[0];
        color != 0 && (y == (img.width() - 1) as i32 || color1 == 0)
    }
}