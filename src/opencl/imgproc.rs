extern crate ocl;
extern crate find_folder;

use self::ocl::{core, util, Device, Context, Queue, Kernel, Program, MemFlags, Buffer};
use self::ocl::builders::ProgramBuilder;
use self::ocl::flags::DeviceType;
use self::find_folder::Search;
use std::mem;

pub fn new(gpu: bool, dim: (u32, u32)) -> Canny {
    let cpu_or_gpu = if gpu { "gpu/" } else { "cpu/" };
    let dtype = if gpu { DeviceType::new().gpu() } else { DeviceType::new().cpu() };
    let workgroup_size = if gpu { 16 } else { 1 };

    let path = Search::ParentsThenKids(3, 3).for_folder("kernels").unwrap();

    let gauss_src_file = path.join(cpu_or_gpu.to_string() + "gaussian_kernel.cl");
    let hyst_src_file = path.join(cpu_or_gpu.to_string() + "hysteresis_kernel.cl");
    let non_max_supp_src_file = path.join(cpu_or_gpu.to_string() + "non_max_suppression_kernel.cl");
    let sobel_src_file = path.join(cpu_or_gpu.to_string() + "sobel_kernel.cl");
    
    let context = Context::builder().devices(Device::specifier().type_flags(dtype).first()).build().unwrap();
    let device = context.devices()[0];

    println!("Running on: {} - {}", device.vendor(), device.name());

    let queue = Queue::new(&context, device, None).unwrap();

    let gauss = Program::builder().src_file(gauss_src_file).build(&context).unwrap();
    let hyst = Program::builder().src_file(hyst_src_file).build(&context).unwrap();
    let nms = Program::builder().src_file(non_max_supp_src_file).build(&context).unwrap();
    let sobel = Program::builder().src_file(sobel_src_file).build(&context).unwrap();

    let pqs = ProcessingProQues {
        gauss_pq: gauss,
        hyst_pq: hyst,
        nms_pq: nms,
        sobel_pq: sobel,
        queue: queue,
        lws: workgroup_size,
    };

    Canny::new(pqs, calculate_dimensions(dim, workgroup_size))
}

fn calculate_dimensions(dim: (u32, u32), workgroup_size: u32) -> (u32, u32) {
    let cols = ((dim.0 - 2) / workgroup_size) * workgroup_size + 2;
    let rows = ((dim.1 - 2) / workgroup_size) * workgroup_size + 2;

    (cols, rows)
}

pub struct ProcessingProQues {
    gauss_pq: Program,
    hyst_pq: Program,
    nms_pq: Program,
    sobel_pq: Program,
    queue: Queue,
    lws: u32,
}

pub struct Canny{
    buffers: Vec<Buffer<u8>>,
    theta_buffer: Buffer<u8>,
    buffer_index: usize,
    proques: ProcessingProQues,
    dim: (u32, u32)
}

impl Canny {
    // add code here

    pub fn new(pqs: ProcessingProQues, dim: (u32, u32)) -> Canny {
        let kdim = dim.0.checked_mul(dim.1).unwrap().checked_mul(mem::size_of::<u8>() as u32).unwrap();
        
        println!("Buffer Size: {}" , kdim);

        let next_buffer = Buffer::builder()
            .queue(pqs.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
            //.host_data(&data)
            .build().unwrap();
        
        let prev_buffer = Buffer::builder()
            .queue(pqs.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
            .build().unwrap();

        let theta_buffer = Buffer::builder()
            .queue(pqs.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
            .build().unwrap();

        let buffers = vec![prev_buffer, next_buffer];

        Canny { buffers: buffers, theta_buffer: theta_buffer, buffer_index: 0, proques: pqs, dim: dim }
    }

    pub fn get_desired_size(&mut self) -> (u32, u32) {
        self.dim
    }

    fn next_buffer(&self) -> &Buffer<u8> {
        &self.buffers[self.buffer_index]
    }

    fn prev_buffer(&self) -> &Buffer<u8> {
        &self.buffers[self.buffer_index ^ 1] //bitwise xor with 1 switches between 0 and 1
    }

    fn advance_buffer(&mut self) -> &Self {
        self.buffer_index = self.buffer_index ^ 1; //bitwise xor with 1 switches between 0 and 1

        self
    }

    fn execute_gaussian(&mut self) {
        println!("Gauss: Next buffer {} - Prev buffer {}", self.buffer_index, self.buffer_index ^ 1);

        let kernel = Kernel::new("gaussian_kernel", &self.proques.gauss_pq).unwrap()
            .queue(self.proques.queue.clone())
            .gwo([1, 1])
            .gws([self.dim.0 - 2, self.dim.1 - 2])
            .lws([self.proques.lws, self.proques.lws])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        println!("Gauss: Kernel global work size: {:?}", kernel.get_gws());

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_sobel(&mut self) {
        println!("Sobel: Next buffer {} - Prev buffer {}", self.buffer_index, self.buffer_index ^ 1);

        let kernel = Kernel::new("sobel_kernel", &self.proques.sobel_pq).unwrap()
            .queue(self.proques.queue.clone())
            .gwo([1, 1])
            .gws([self.dim.0 - 2, self.dim.1 - 2])
            .lws([self.proques.lws, self.proques.lws])
            .arg_buf_named("data", Some(self.prev_buffer()))
            .arg_buf_named("out", Some(self.next_buffer()))
            .arg_buf_named("theta", Some(&self.theta_buffer))
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        println!("Sobel: Kernel global work size: {:?}", kernel.get_gws());

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_nms(&mut self) {
        println!("NMS: Next buffer {} - Prev buffer {}", self.buffer_index, self.buffer_index ^ 1);

        let kernel = Kernel::new("non_max_suppression_kernel", &self.proques.nms_pq).unwrap()
            .queue(self.proques.queue.clone())
            .gwo([1, 1])
            .gws([self.dim.0 - 2, self.dim.1 - 2])
            .lws([self.proques.lws, self.proques.lws])
            .arg_buf_named("data", Some(self.prev_buffer()))
            .arg_buf_named("out", Some(self.next_buffer()))
            .arg_buf_named("theta", Some(&self.theta_buffer))
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        println!("NMS: Kernel global work size: {:?}", kernel.get_gws());

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_hyst(&mut self) {
        println!("Hysteresis: Next buffer {} - Prev buffer {}", self.buffer_index, self.buffer_index ^ 1);

        let kernel = Kernel::new("hysteresis_kernel", &self.proques.hyst_pq).unwrap()
            .queue(self.proques.queue.clone())
            .gwo([1, 1])
            .gws([self.dim.0 - 2, self.dim.1 - 2])
            .lws([self.proques.lws, self.proques.lws])
            .arg_buf_named("data", Some(self.prev_buffer()))
            .arg_buf_named("out", Some(self.next_buffer()))
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        println!("Hysteresis: Kernel global work size: {:?}", kernel.get_gws());

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    pub fn execute_edge_detection(&mut self, data: Vec<u8>) -> Vec<u8> {
        //Upload to old buffer
        self.prev_buffer().cmd().write(&data).enq().unwrap();

        println!("Beginning: Next buffer {} - Prev buffer {}", self.buffer_index, self.buffer_index ^ 1);

        //self.execute_test();
        self.execute_gaussian();

        //self.execute_sobel();

       // self.execute_nms();

       // self.execute_hyst();

        println!("End: Next buffer {} - Prev buffer {}", self.buffer_index, self.buffer_index ^ 1);

        let mut res = vec![0u8; (self.dim.0 * self.dim.1) as usize]; //pretty unsafe
        self.prev_buffer().read(&mut res).enq().unwrap();

        res
    }
}