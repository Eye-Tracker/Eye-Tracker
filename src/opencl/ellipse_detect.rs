use ocl::{Queue, Kernel, Context, Program, MemFlags, Buffer};
use std::mem;
use std::path::PathBuf;

#[macro_use]
use opencl;

pub struct EllipseRANSAC {
    ellipse_detect: Program,
}

impl EllipseRANSAC {
    pub fn new(path: PathBuf, context: &Context, queue: Queue) -> EllipseRANSAC {
        let ellipse_detect = Program::builder().src_file(path.join("RANSAC_circle_finder.cl")).build(context).unwrap();

        EllipseRANSAC { ellipse_detect: ellipse_detect }
    }
}