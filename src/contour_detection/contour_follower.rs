use image::GrayImage;
use contour_detection::Coordinates;
use contour_detection;

pub trait FollowingStrategy {
    fn contour(&self, img: &GrayImage, ij: Coordinates, i2j2: Coordinates) -> Vec<Coordinates> {
        let mut ret =Vec::new();
        self.do_contouring(img, ij, i2j2, |c| ret.push(c));
        return ret;
    }

    fn do_contouring<F>(&self, img: &GrayImage, ij: Coordinates, i2j2: Coordinates, 
                            func: F) -> () where F: FnMut(Coordinates) -> (); 
}

pub struct SuzukiStrategy;

impl SuzukiStrategy {
    pub fn directed_contour<F>(&self, img: &GrayImage, ij: Coordinates, i2j2: Coordinates, 
                            mut func: F) -> () where F: FnMut((Coordinates, [bool; 8])) -> () {
        let mut dir = contour_detection::direction::fromTo(ij, i2j2).unwrap();
        let mut trace = dir.clockwise();
        let mut i1j1: Option<Coordinates> = None;

        while trace != dir {
            let active_pixel = trace.active(ij, &img);
            if let Ok(res) = active_pixel {
                if let Some(pixel) = res {
                    i1j1 = Some(pixel);
                    break;
                }
            }
            trace = trace.clockwise();
        }
        if i1j1 == None {
            return;
        }

        let mut i2j2 = i1j1.unwrap(); //This can't be None anymore
        let mut i3j3 = ij;
        let mut checked = [false, false, false, false, false, false, false, false]; //N , NE ,E ,SE ,S ,SW ,W ,NW
        
        fn reset_checked(arr: &mut [bool; 8]) {
            for i in 0..arr.len() {
                arr[i] = false;
            }
        }
        
        loop {
            dir = contour_detection::direction::fromTo(i3j3, i2j2).unwrap();
            trace = dir.counter_clockwise();
            let mut i4j4: Option<Coordinates> = None;
            reset_checked(&mut checked); //TODO test if this really resets
            loop {
                i4j4 = if let Ok(pix) = trace.active(i3j3, &img) {
                    pix
                } else {
                    None
                };
                if i4j4 != None {
                    break;
                }
                checked[trace.as_value() as usize] = true;
                trace = trace.counter_clockwise();
            }

            func((i3j3, checked));
            if i4j4.unwrap() == ij && i3j3 == i1j1.unwrap() {
                break;
            }
            i2j2 = i3j3;
            i3j3 = i4j4.unwrap();
        }
    }
}
impl FollowingStrategy for SuzukiStrategy {
    fn do_contouring<F>(&self, img: &GrayImage, ij: Coordinates, i2j2: Coordinates, 
                            mut func: F) -> () where F: FnMut(Coordinates) -> () {
        self.directed_contour(img, ij, i2j2, |(c, _)| func(c));
    }
}