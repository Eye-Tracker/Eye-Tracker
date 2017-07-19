use ocl::{Queue, Kernel, Context, Program, MemFlags, Buffer};
use ocl::prm::Int3;
use std::mem;
use std::path::PathBuf;
use contour_detection::contour::Contour;
use contour_detection::shape::{Polygon, Points, PointList};
use rand;

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

    fn launch_kernel(&self, consens_x: &[usize], consens_y: &[usize], consens_size: &[usize], max_width: usize, num_contours: usize) -> Vec<RansacResult> {
        let mut rands = vec![Int3::new(0,0,0); self.params.num_iterations as usize * num_contours * 10usize];
        //Generate random numbers on CPU
        for i in 0..rands.len() {
            let rand = rand::random::<(i32, i32, i32)>();
            rands[i] = Int3::new(rand.0, rand.1, rand.2);
        }

        let consens_x_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(consens_x.len())
            .host_data(consens_x)
            .build().unwrap();
        
        let consens_y_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(consens_y.len())
            .host_data(consens_y)
            .build().unwrap();
        
        let consens_size_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(consens_size.len())
            .host_data(consens_size)
            .build().unwrap();

        let rand_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(rands.len())
            .host_data(&rands)
            .build().unwrap();

        let mut result_center = vec![0; num_contours * 2];
        let mut result_radius = vec![0.0; num_contours];

        //TODO not quite sure with the len here
        let result_center_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(num_contours * 2)
            .host_data(&result_center)
            .build().unwrap();

        let result_radius_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(num_contours)
            .host_data(&result_radius)
            .build().unwrap();

        let kernel = Kernel::new("ransac_kernel", &self.ellipse_detect).unwrap()
            .queue(self.queue.clone())
            .gws(num_contours)
            .lws(self.params.num_iterations)
            .arg_buf(&consens_x_buffer)
            .arg_buf(&consens_y_buffer)
            .arg_buf(&consens_size_buffer)
            .arg_buf(&result_center_buffer)
            .arg_buf(&result_radius_buffer)
            .arg_buf(&rand_buffer)
            .arg_scl(self.params.max_point_seperation)
            .arg_scl(self.params.min_point_seperation)
            .arg_scl(self.params.colinear_tolerance)
            .arg_scl(self.params.radius_tolerance)
            .arg_scl(self.params.point_threshold)
            .arg_scl(self.params.num_iterations);

        kernel.enq().unwrap();

        result_center_buffer.read(&mut result_center).enq().unwrap();
        result_radius_buffer.read(&mut result_radius).enq().unwrap();

        let mut result = Vec::with_capacity(result_radius.len());
        for i in 0..result.len() {
            result.push((result_center[i*2], result_center[i*2+1], result_radius[i]));
        }
        result
    }

    pub fn execute_ellipse_fit(&self, contours: &[Contour]) -> Option<Vec<RansacResult>> {
        if contours.len() > 0 && contours.len() < 200 { //Fitting over 200 contours is too expensive
            let mut max_len = 0;
            for c in contours {
                let len = c.points.get_vertices().len();
                if max_len < len {
                    max_len = len;
                }
            }

            let dim = contours.len() * max_len;
            let mut consens_x = vec![0; contours.len() * max_len];
            let mut consens_y = vec![0; contours.len() * max_len];
            let mut consens_size = vec![0; contours.len()];

            for i in 0..contours.len() {
                let points = contours[i].points.get_vertices();
                let len = points.len();
                for j in 0..len {
                    consens_x[i * max_len + j] = points[j].x;
                    consens_y[i * max_len + j] = points[j].y;
                }

                consens_size[i] = len;
            }

            return Some(self.launch_kernel(&consens_x, &consens_y, &consens_size, max_len, contours.len()));
        }

         None
    }
}