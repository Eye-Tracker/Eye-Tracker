use ocl::{Queue, Kernel, Context, Program, MemFlags, Buffer};
use ocl::prm::{Float, Float2, Int, Int2, Int3};
use ocl::SpatialDims;
use std::mem;
use std::path::PathBuf;
use contour_detection::contour::Contour;
use contour_detection::shape::{Polygon, Points, PointList};
use rand;
use rand::distributions::{IndependentSample, Range};

#[macro_use]
use opencl;

struct RansacParams {
    max_point_seperation: f32,
    min_point_seperation: f32,
    colinear_tolerance: f32,
    radius_tolerance: f32,
    point_threshold: f32,
    circle_threshold: f32,
    num_iterations: i32,
}

pub struct EllipseRANSAC {
    ellipse_detect: Program,
    queue: Queue,
    params: RansacParams,
}

type RansacResult = (usize, usize, f32);
impl EllipseRANSAC {
    pub fn new(circle_threshold: f32, num_iterations: i32, width: usize, height: usize, path: PathBuf, context: &Context, queue: Queue) -> EllipseRANSAC {
        let ellipse_detect = Program::builder().src_file(path.join("RANSAC_circle_finder.cl")).build(context).unwrap();

        let params = RansacParams{
            max_point_seperation: height as f32,
            min_point_seperation: height as f32 / 50f32,
            colinear_tolerance: 1f32,
            radius_tolerance: height as f32 / 2f32,
            point_threshold: 10f32,
            circle_threshold: circle_threshold,
            num_iterations: num_iterations,
        };

        EllipseRANSAC { ellipse_detect: ellipse_detect, queue: queue, params: params }
    }

    fn launch_kernel(&self, consens_x: Vec<i32>, consens_y: Vec<i32>, consens_size: Vec<i32>, max_width: i32, num_contours: i32) -> Vec<RansacResult> {
        let mut rands1 = vec![0i32; self.params.num_iterations as usize * num_contours as usize * 10usize];
        let mut rands2 = vec![0i32; self.params.num_iterations as usize * num_contours as usize * 10usize];
        let mut rands3 = vec![0i32; self.params.num_iterations as usize * num_contours as usize * 10usize];
        //Generate random numbers on CPU
        let between = Range::new(0, max_width);
        let mut rng = rand::thread_rng();
        for i in 0..rands1.len() {
            rands1[i] = between.ind_sample(&mut rng);
            rands2[i] = between.ind_sample(&mut rng);
            rands3[i] = between.ind_sample(&mut rng);
        }

        let consens_x_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(SpatialDims::One(consens_x.len()))
            .host_data(&consens_x)
            .build().expect("Consens x buffer wasn't created");
        
        let consens_y_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(SpatialDims::One(consens_y.len()))
            .host_data(&consens_y)
            .build().expect("Consens y buffer wasn't created");
        
        let consens_size_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(SpatialDims::One(consens_size.len()))
            .host_data(&consens_size)
            .build().expect("Consens Size buffer wasn't created");

        let rand1_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(SpatialDims::One(rands1.len()))
            .host_data(&rands1)
            .build().expect("Random number 1 buffer wasn't created");

        let rand2_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(SpatialDims::One(rands2.len()))
            .host_data(&rands2)
            .build().expect("Random number 2 buffer wasn't created");

        let rand3_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(SpatialDims::One(rands3.len()))
            .host_data(&rands3)
            .build().expect("Random number 3 buffer wasn't created");

        let result_cx_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(SpatialDims::One(num_contours as usize))
            .build().expect("Cx buffer wasn't created");

        let result_cy_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(SpatialDims::One(num_contours as usize))
            .build().expect("Cy buffer wasn't created");

        let result_radius_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(SpatialDims::One(num_contours as usize))
            .build().expect("Radius buffer wasn't created");

        let kernel = Kernel::new("ransac_kernel", &self.ellipse_detect).unwrap()
            .queue(self.queue.clone())
            .gws([num_contours as usize * self.params.num_iterations as usize])
            .lws([self.params.num_iterations as usize])
            .arg_buf(&consens_x_buffer)
            .arg_buf(&consens_y_buffer)
            .arg_buf(&consens_size_buffer)
            .arg_buf(&result_cx_buffer)
            .arg_buf(&result_cy_buffer)
            .arg_buf(&result_radius_buffer)
            .arg_buf(&rand1_buffer)
            .arg_buf(&rand2_buffer)
            .arg_buf(&rand3_buffer)
            .arg_loc::<Int>(self.params.num_iterations as usize * mem::size_of::<Int>())
            .arg_loc::<Int2>(self.params.num_iterations as usize * mem::size_of::<Int2>())
            .arg_loc::<Float>(self.params.num_iterations as usize * mem::size_of::<Float>())
            .arg_vec(Float2::new(self.params.min_point_seperation, self.params.max_point_seperation))
            .arg_vec(Float2::new(self.params.colinear_tolerance, self.params.radius_tolerance))
            .arg_vec(Int2::new(self.params.num_iterations, max_width));

        kernel.enq().expect("Kernel couldn't be started");

        let mut result_cx = vec![0i32; num_contours as usize]; 
        let mut result_cy = vec![0i32; num_contours as usize]; 
        let mut result_radius = vec![0f32; num_contours as usize];

        result_cx_buffer.read(&mut result_cx).enq().unwrap();
        result_cy_buffer.read(&mut result_cy).enq().unwrap();
        result_radius_buffer.read(&mut result_radius).enq().unwrap();

        let mut result = Vec::with_capacity(result_radius.len());
        for i in 0..result_radius.len() {
            let cx = result_cx[i];
            let cy = result_cy[i];
            let radius = result_radius[i];
            result.push((cx as usize, cy as usize, radius));
        }

        result
    }

    pub fn execute_ellipse_fit(&self, contours: &[Contour]) -> Option<Vec<RansacResult>> {
        if contours.len() > 2 && contours.len() < 2000 { //Fitting over 2000 contours is too expensive
            let mut max_len = 0;
            for c in contours {
                let len = c.points.get_vertices().len();
                if max_len < len {
                    max_len = len;
                }
            }

            let dim = contours.len() * max_len;
            let mut consens_x = vec![0i32; dim];
            let mut consens_y = vec![0i32; dim];
            let mut consens_size = vec![0i32; contours.len()];

            for y in 0..contours.len() {
                let points = contours[y].points.get_vertices();
                let len = points.len();
                for x in 0..len {
                    consens_x[y * max_len + x] = points[x].x as i32;
                    consens_y[y * max_len + x] = points[x].y as i32;
                }

                consens_size[y] = len as i32;
            }

            return Some(self.launch_kernel(consens_x, consens_y, consens_size, max_len as i32, contours.len() as i32));
        }

        None
    }
}