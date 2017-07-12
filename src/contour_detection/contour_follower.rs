use image::RgbaImage;
use contour_detection::Coordinates;
use contour_detection::direction::Direction;
use contour_detection;
use std;

pub trait FollowingStrategy {
    fn contour(&self, img: RgbaImage, start: Coordinates, from: Coordinates) -> Vec<Coordinates> {
        let mut ret =Vec::new();
        self.do_contouring(img, start, from, |c| ret.push(c));
        return ret;
    }

    fn do_contouring<F>(&self, img: RgbaImage, start: Coordinates, from: Coordinates, 
                            func: F) -> () where F: FnMut(Coordinates) -> (); 
}

struct SuzukiStrategy;

impl FollowingStrategy for SuzukiStrategy {
    fn do_contouring<F>(&self, img: RgbaImage, start: Coordinates, from: Coordinates, 
                            mut func: F) -> () where F: FnMut(Coordinates) -> () {
        let mut dir = contour_detection::direction::fromTo(start, from).unwrap();
        let mut trace = dir.clockwise();
        let mut new_active: Option<Coordinates> = None;

        while trace != dir {
            let active_pixel = trace.active(start, &img);
            if let Ok(res) = active_pixel {
                if let Some(pixel) = res {
                    new_active = Some(pixel);
                    break;
                }
            }
            trace = trace.clockwise();
        }
        if new_active == None {
            return;
        }

        let mut temp = new_active.unwrap(); //This can't be None anymore
        let mut temp2 = start;
        let mut checked = vec![false, false, false, false, false, false, false, false]; //N , NE ,E ,SE ,S ,SW ,W ,NW
        
        fn reset_checked(arr: &mut Vec<bool>) {
            for i in 0..arr.len() {
                arr[i] = false;
            }
        }
        
        loop {
            dir = contour_detection::direction::fromTo(temp2, temp).unwrap();
            trace = dir.counter_clockwise();
            let mut temp3: Option<Coordinates> = None;
            reset_checked(&mut checked); //TODO test if this really resets
            loop {
                temp3 = if let Ok(pix) = trace.active(temp2, &img) {
                    pix
                } else {
                    None
                };
                if temp3 != None {
                    break;
                }
                checked[trace.as_value() as usize] = true;
                trace = trace.counter_clockwise();
            }

            func(temp2);
            if temp3.unwrap() == start && temp2 == from {
                break;
            }
            temp = temp2;
            temp2 = temp3.unwrap();
        }

    }
}