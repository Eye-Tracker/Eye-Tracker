use ocl::{Device, Context, Queue, Kernel, Program, MemFlags, Buffer};
use ocl::flags::DeviceType;
use find_folder::Search;
use std::mem;

pub fn new(gpu: bool, dim: (u32, u32)) -> Canny {
    let cpu_or_gpu = if gpu { "gpu/" } else { "cpu/" };
    let dtype = if gpu { DeviceType::new().gpu() } else { DeviceType::new().cpu() };
    //let workgroup_size = if gpu { 16 } else { 1 };

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
        gauss: gauss,
        hyst: hyst,
        nms: nms,
        sobel: sobel,
        queue: queue,
    };

    Canny::new(pqs, dim)
}

pub struct ProcessingProQues {
    gauss: Program,
    hyst: Program,
    nms: Program,
    sobel: Program,
    queue: Queue,
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

        let next_buffer = Buffer::builder()
            .queue(pqs.queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
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
        let kernel = Kernel::new("gaussian_kernel", &self.proques.gauss).unwrap()
            .queue(self.proques.queue.clone())
            .gws([self.dim.0, self.dim.1])
            //.lws([self.proques.lws, self.proques.lws])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_sobel(&mut self) {
        let kernel = Kernel::new("sobel_kernel", &self.proques.sobel).unwrap()
            .queue(self.proques.queue.clone())
            .gws([self.dim.0, self.dim.1])
           // .lws([self.proques.lws, self.proques.lws])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_buf(&self.theta_buffer)
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_nms(&mut self) {
        let kernel = Kernel::new("non_max_suppression_kernel", &self.proques.nms).unwrap()
            .queue(self.proques.queue.clone())
            .gws([self.dim.0 , self.dim.1])
            //.lws([self.proques.lws, self.proques.lws])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_buf(&self.theta_buffer)
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_hyst(&mut self) {
        let kernel = Kernel::new("hysteresis_kernel", &self.proques.hyst).unwrap()
            .queue(self.proques.queue.clone())
            .gws([self.dim.0, self.dim.1])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    pub fn execute_edge_detection(&mut self, data: Vec<u8>) -> Vec<u8> {
        //Upload to old buffer
        self.prev_buffer().cmd().write(&data).enq().unwrap(); 

        self.execute_gaussian();

        self.execute_sobel();

        self.execute_nms();

        self.execute_hyst();

        let mut res = vec![0u8; self.prev_buffer().len()];
        self.prev_buffer().read(&mut res).enq().unwrap();

        res
    }
}