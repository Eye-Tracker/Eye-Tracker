use ocl::{Queue, Kernel, Context, Program, MemFlags, Buffer};
use std::mem;
use std::path::PathBuf;

struct Programms {
    gauss: Program,
    hyst: Program,
    nms: Program,
    sobel: Program,
}

pub struct Canny{
    buffers: Vec<Buffer<u8>>,
    theta_buffer: Buffer<u8>,
    buffer_index: usize,
    programms: Programms,
    queue: Queue,
    dim: (u32, u32),
}

impl Canny {
    pub fn new(path: PathBuf, context: &Context, queue: Queue, dim: (u32, u32)) -> Canny {
        let gauss = Program::builder().src_file(path.join("gaussian_kernel.cl")).build(context).unwrap();
        let hyst = Program::builder().src_file(path.join("hysteresis_kernel.cl")).build(context).unwrap();
        let nms = Program::builder().src_file(path.join("non_max_suppression_kernel.cl")).build(context).unwrap();
        let sobel = Program::builder().src_file(path.join("sobel_kernel.cl")).build(context).unwrap();

        let programms = Programms {
            gauss: gauss,
            hyst: hyst,
            nms: nms,
            sobel: sobel
        };

        let kdim = dim.0.checked_mul(dim.1).unwrap().checked_mul(mem::size_of::<u8>() as u32).unwrap();

        let next_buffer = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
            .build().unwrap();
        
        let prev_buffer = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
            .build().unwrap();

        let theta_buffer = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_write().alloc_host_ptr())
            .dims(kdim.clone())
            .build().unwrap();

        let buffers = vec![prev_buffer, next_buffer];

        Canny { buffers: buffers, theta_buffer: theta_buffer, buffer_index: 0, programms: programms, queue: queue, dim: dim }
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
        let kernel = Kernel::new("gaussian_kernel", &self.programms.gauss).unwrap()
            .queue(self.queue.clone())
            .gws([self.dim.0, self.dim.1])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_sobel(&mut self) {
        let kernel = Kernel::new("sobel_kernel", &self.programms.sobel).unwrap()
            .queue(self.queue.clone())
            .gws([self.dim.0, self.dim.1])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_buf(&self.theta_buffer)
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_nms(&mut self) {
        let kernel = Kernel::new("non_max_suppression_kernel", &self.programms.nms).unwrap()
            .queue(self.queue.clone())
            .gws([self.dim.0 , self.dim.1])
            .arg_buf(self.prev_buffer())
            .arg_buf(self.next_buffer())
            .arg_buf(&self.theta_buffer)
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();

        self.advance_buffer();
    }

    fn execute_hyst(&mut self) {
        let kernel = Kernel::new("hysteresis_kernel", &self.programms.hyst).unwrap()
            .queue(self.queue.clone())
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