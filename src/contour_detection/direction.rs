use image::GrayImage;
use contour_detection::Coordinates;
use std;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(i32)]
pub enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

static DIR_X: &'static [i32] = &[0, 1, 1, 1, 0, -1, -1, -1];
static DIR_Y: &'static [i32] = &[-1, -1, 0, 1, 1, 1, 0, -1];
impl Direction {
    pub fn as_value(&self) -> i32 {
        *self as i32
    }

    pub fn clockwise(&self) -> Direction {
        let val = *self as i32;
        unsafe{ 
            let ret: Direction = std::mem::transmute((val + 1) % 8);
            return ret; 
        } //This is really unsafe
        panic!("Unsafe block in Clockwise Direction failed!");
    }

    pub fn counter_clockwise(&self) -> Direction {
        let val = *self as i32;
        let desired = if val - 1 == -1 {
            8 - 1
        } else {
            val - 1
        };
        unsafe { 
            let ret: Direction = std::mem::transmute(desired);
            return ret;
        } //This is really unsafe
        panic!("Unsafe block in Counter Clockwise Direction failed!");
    }

    pub fn active(&self, pos: Coordinates, img: &GrayImage) -> Result<Option<Coordinates>, &'static str> {
        let cur = *self as i32;
        let y = pos.y as i32 + DIR_Y[cur as usize];
        let x = pos.x as i32 + DIR_X[cur as usize];
        if x < 0 || x >= img.width() as i32 || y < 0 || y >= img.height() as i32 {
            Err("Position needs to be within the image bounds!")
        } else {
            let color: u8 = img.get_pixel(x as u32, y as u32).data[0];
            if color != 0 {
                Ok(Some(Coordinates::new(x as usize, y as usize)))
            } else {
                Ok(None)
            }
        }
    }
}

pub fn fromTo(from: Coordinates, to: Coordinates) -> Result<Direction, &'static str> {
    if from == to {
        Err("From and To positions are the same!")
    } else {
        Ok(if from.eq_y(&to) {
            if from.lt_x(&to) {
                Direction::East
            } else {
                Direction::West
            }
        } else if from.lt_y(&to) {
            if from.eq_x(&to) {
                Direction::South
            } else if from.lt_x(&to) {
                Direction::SouthEast
            } else {
                Direction::SouthWest
            }
        } else {
            if from.eq_x(&to) {
                Direction::North
            } else if from.lt_x(&to) {
                Direction::NorthEast
            } else {
                Direction::NorthWest
            }
        })
    }
}
