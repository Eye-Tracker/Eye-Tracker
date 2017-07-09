use image::RgbaImage;

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
    fn eq(&self, other: Coordinates) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Coordinates {}

#[derive(Debug)]
#[repr(i32)]
enum Direction {
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
static ENTRY: &'static [Direction] = &[Direction::West, Direction::West,
    Direction::North, Direction::North, Direction::East, Direction::East,
    Direction::South, Direction::South];
static CCENTRY: &'static [Direction] = &[Direction::East, Direction::South,
    Direction::South, Direction::West, Direction::West, Direction::North,
    Direction::North, Direction::East];
impl Direction {
    pub fn clockwise(&self) -> Direction {
        let val = self as i32;
        let ret: Direction = unsafe{ std::mem::transmute((val + 1) % 8) } //This is really unsafe
        ret
    }

    pub fn counter_clockwise(&self) -> Direction {
        let val = self as i32;
        let desired = if val - 1 == -1 {
            8 - 1
        } else {
            val - 1
        }
        let ret: Direction = unsafe{ std::mem::transmute(desired) } //This is really unsafe
        ret
    }

    pub fn active(&self, pos: Coordinates, img: RgbaImage) -> Result<Option<Coordinates>, &'static str> {
        let cur = self as i32;
        let y = pos.y + DIR_Y[cur];
        let x = pos.x + DIR_X[cur];
        if x < 0 || x >= img.width() || y < 0 || y >= img.height() {
            Err("Position needs to be within the image bounds!")
        }
        let color: u8 = img.get_pixel(x, y).channels().0;
        if color != 0 {
            Ok(Some(Coordinates::new(x, y)))
        } else {
            Ok(None)
        }
    }

    pub fn new_position(&self, pos: Coordinates) -> Coordinates {
        let cur = self as i32;
        let y = pos.y + DIR_Y[cur];
        let x = pos.x + DIR_X[cur];
        Coordinates::new(x, y)
    }

    pub fn clockwise_entry_direction(&self) -> Direction {
        let val = self as i32;
        let ret: ENTRY = unsafe{ std::mem::transmute(val) } //This is really unsafe
        ret
    }

    pub fn counter_clockwise_entry_direction(&self) -> Direction {
        let val = self as i32;
        let ret: CCENTRY = unsafe{ std::mem::transmute(val) } //This is really unsafe
        ret
    }
}

fn fromTo(from: Coordinates, to: Coordinates) -> Result<Direction, &'static str> {
    if (from == to) {
        Err("From and To positions are the same!")
    }
    Ok(if from.eq_y(to) {
        if from.lt_x(to) {
            Direction::East
        } else {
            Direction::West
        }
    } else if from.lt_y(to) {
        if from.eq_x(to) {
            Direction::South
        } else if (from.lt_x(to)) {
            Direction::SouthEast
        } else {
            Direction::SouthWest
        }
    } else {
        if from.eq_x(to) {
            Direction::North
        } else if from.lt_x(to) {
            Direction::NorthEast
        } else {
            Direction::NorthWest
        }
    })
}
