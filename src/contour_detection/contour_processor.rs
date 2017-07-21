use contour_detection::contour::{Contour, ContourBuilder};
use contour_detection::shape::{Rectangle, Shape, Points};
use contour_detection::contour_follower::SuzukiStrategy;
use contour_detection::{ContourType, Coordinates};
use contour_detection::direction;
use contour_detection::FloatImage;
use image::GrayImage;
use indextree::Arena;
use indextree;
use std::collections::HashMap;
use ordered_float::*;

pub struct ContourProcessor;

impl ContourProcessor {
    pub fn find_contours(&self, img: &GrayImage, min_relative_child_prop: f64) -> Vec<Contour> {
        let image = FloatImage::new(img);
        
        let bounds = Rectangle::new(0, 0, (img.width() - 1) as usize, (img.height() - 1) as usize);
        
        let arena = &mut Arena::<Contour>::new();

        let root = arena.new_node(
            ContourBuilder::new()
                .set_type(ContourType::Hole)
                .set_points(bounds.as_polygon().get_vertices()).finish());
        
        let border_follow = SuzukiStrategy;

        let mut nbd = NotNaN::new(1f32).unwrap();
        let mut lnbd =  NotNaN::new(1f32).unwrap();

        let mut border_map = HashMap::new();
        border_map.insert(lnbd, root);

        for y in 0..img.height() {
            lnbd = NotNaN::new(1f32).unwrap();
            for x in 0..img.width() {
                let cur_pixel = image.get_pixel(x, y);
                let is_outer = self.is_outer_border_start(&image, x as i32, y as i32);
                let is_hole = self.is_hole_border_start(&image, x as i32, y as i32);
                if is_outer || is_hole {
                    let mut from = Coordinates::new(x as usize, y as usize);
                    let mut border = ContourBuilder::new().set_start(from);

                    let start = Coordinates::new(x as usize, y as usize);

                    if is_outer {
                        nbd = nbd + NotNaN::new(1f32).unwrap();
                        from.x = from.x - 1;
                        border = border.set_type(ContourType::Outer);
                    } else {
                        nbd = nbd + NotNaN::new(1f32).unwrap();
                        if cur_pixel > 1f32 {
                            lnbd = NotNaN::new(cur_pixel).unwrap();
                        }
                        from.x = from.x + 1;
                        border = border.set_type(ContourType::Hole);
                    }

                    let mut points = Vec::new();
                    border_follow.directed_contour(&image, start, from, |(coord, checked)| {
                        points.push(coord);
                        if self.crosses_east_border(&image, checked, coord) {
                            image.set_pixel(coord.x as u32, coord.y as u32, -nbd.into_inner());
                        } else if image.get_pixel(coord.x as u32, coord.y as u32) == 1f32 {
                             image.set_pixel(coord.x as u32, coord.y as u32, nbd.into_inner());
                        }
                    });

                    if points.len() == 0 {
                        points.push(start);
                        image.set_pixel(x, y, -nbd.into_inner());
                    }

                    let contour = arena.new_node(border.set_points(points).finish());
                    if is_outer {
                        if let Some(border_prime) = border_map.get(&lnbd) {
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
                    } else {
                        if let Some(border_prime) = border_map.get(&lnbd) {
                            match arena[border_prime.clone()].data.ctype {
                                Some(ContourType::Hole) => {
                                    let p_parent = border_prime.ancestors(arena).next().unwrap();
                                    p_parent.append(contour, arena);
                                    
                                },
                                Some(ContourType::Outer) => {
                                    border_prime.append(contour, arena);
                                },
                                None => panic!("Type should always been set!"),
                            }
                        }
                    }

                    border_map.insert(nbd, contour);
                }
                if cur_pixel != 0f32 && cur_pixel != 1f32 {
                    lnbd = NotNaN::new(cur_pixel.abs()).unwrap();
                }
            }
        }

        if min_relative_child_prop > 0f64 {
            self.remove_small(arena, &root, min_relative_child_prop);
        }
        

        root.traverse(arena).map(|ne| {
                match ne {
                    indextree::NodeEdge::Start(val) => arena[val].clone().data,
                    indextree::NodeEdge::End(val) => arena[val].clone().data,
                }
            }).collect::<Vec<Contour>>()
    }

    fn crosses_east_border(&self, img: &FloatImage, checked: [bool; 8], p: Coordinates) -> bool {
        let b = checked[direction::from_to(p, Coordinates::new(p.x + 1, p.y)).unwrap().as_value() as usize];
        img.get_pixel(p.x as u32, p.y as u32) != 0f32 && (p.x == (img.width() - 1) as usize || b)
    }

    fn is_outer_border_start(&self, img: &FloatImage, x: i32, y: i32) -> bool {
        let color: f32 = img.get_pixel(x as u32, y as u32);
        let t = if y == 0 { y + 1} else { y };
        let color1: f32 = img.get_pixel(x as u32, (t - 1) as u32);
        color == 1f32 && (y == 0 || color1 == 0f32)
    }

    fn is_hole_border_start(&self, img: &FloatImage, x: i32, y: i32) -> bool {
        let color: f32 = img.get_pixel(x as u32, y as u32);
        let t = if y == img.height() as i32 - 1 { y - 1} else { y };
        let color1: f32 = img.get_pixel(x as u32, (t + 1) as u32);
        color >= 1f32 && (y == (img.width() - 1) as i32 || color1 == 0f32)
    }

    fn remove_small(&self, arena: &mut Arena<Contour>, root: &indextree::NodeId, min_relative_child_prop: f64) {
        let mut list = Vec::new();
        list.push(root.clone());
        while list.len() != 0 {
            let ret = list.remove(0);
            if let Some(par) = ret.ancestors(arena).nth(1) {
                if arena[ret].data.bounds.calculate_area() / arena[par].data.bounds.calculate_area() < min_relative_child_prop {
                    ret.detach(arena);
                }
            } else {
                let mut childs: Vec<indextree::NodeId> = ret.children(arena).collect();
                list.append(&mut childs);
            }
        }
    }
}