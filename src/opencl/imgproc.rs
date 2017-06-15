//16 threads for gpu. 1 for cpu
extern crate ocl;
extern crate find_folder;

use self::ocl::{core, util, Device, Context, Queue, Program, Buffer};
use self::find_folder::Search;
use std::mem;

pub struct ImgPrc {
    gauss_prog: Program,
    hysteresis_prog: Program,
    non_max_supp_prog: Program,
    sobel_prog: Program,
    queue: Queue,
    workgroup_size: u32,
}

impl ImgPrc {
    pub fn new(gpu: bool) -> Self {
        let cpu_or_gpu = if gpu { "gpu/" } else { "cpu/" };
        let path = Search::ParentsThenKids(3, 3).for_folder("kernels").unwrap();
        let gauss_src_file = path.join(cpu_or_gpu.to_string() + "gaussian_kernel.cl");
        let hyst_src_file = path.join(cpu_or_gpu.to_string() + "hysteresis_kernel.cl");
        let non_max_supp_src_file = path.join(cpu_or_gpu.to_string() + "non_max_suppression_kernel.cl");
        let sobel_src_file = path.join(cpu_or_gpu.to_string() + "sobel_kernel.cl");

        let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
        let device = context.devices()[0];
        let queue = Queue::new(&context, device, None).unwrap();
        let workgroup_size = if gpu { 16 } else { 1 };

        let gauss = Program::builder().src_file(gauss_src_file).devices(device).build(&context).unwrap();
        let hyst = Program::builder().src_file(hyst_src_file).devices(device).build(&context).unwrap();
        let nms = Program::builder().src_file(non_max_supp_src_file).devices(device).build(&context).unwrap();
        let sobel = Program::builder().src_file(sobel_src_file).devices(device).build(&context).unwrap();

        ImgPrc{ 
            gauss_prog: gauss,
            hysteresis_prog: hyst,
            non_max_supp_prog: nms,
            sobel_prog: sobel,
            queue: queue,
            workgroup_size:workgroup_size,
         }
    }

// dim: width, height
    pub fn process_image(&self, data: Vec<u8>, dim: (u32, u32)) {
        // Border extension / mirroring isn't implemented. So don't process edge pixels.
        let rows = ((dim.0 - 2) / self.workgroup_size) * self.workgroup_size + 2;
        let cols = ((dim.1 - 2) / self.workgroup_size) * self.workgroup_size + 2;

        let dim = rows.checked_mul(cols).unwrap().checked_mul(mem::size_of::<u8>() as u32);

        Canny::new(&self.queue, data, dim.unwrap());
    }


}

struct Canny{
    buffers: Vec<Buffer<u8>>,
    theta_buffer: Buffer<u8>,
    buffer_index: u8,
}

impl Canny {
    // add code here

    pub fn new(queue: &Queue, data: Vec<u8>, dim: u32) -> Self {
        let prevBuffer = Buffer::builder()
            .queue(queue.clone())
            .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
            .dims(dim)
            .host_data(&data)
            .build().unwrap();
        
        let nextBuffer = Buffer::builder()
            .queue(queue.clone())
            .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
            .dims(dim)
            .host_data(&data)
            .build().unwrap();

        let thetaBuffer = Buffer::builder()
            .queue(queue.clone())
            .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
            .dims(dim)
            .host_data(&data)
            .build().unwrap();

        let buffers = vec![prevBuffer, nextBuffer];

        Canny { buffers: buffers, theta_buffer: thetaBuffer, buffer_index: 0 }
    }
}