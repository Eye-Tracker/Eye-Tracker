//16 threads for gpu. 1 for cpu
extern crate ocl;
extern crate find_folder;

use self::ocl::{core, util, Device, Context, ProQue, Kernel, Program, Buffer};
use self::ocl::builders::ProgramBuilder;
use self::find_folder::Search;
use std::mem;

pub struct ImgPrc {
    gauss: ProgramBuilder,
    hysteresis: ProgramBuilder,
    non_max_supp: ProgramBuilder,
    sobel: ProgramBuilder,
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
        
        let workgroup_size = if gpu { 16 } else { 1 };

        let gauss = Program::builder().src_file(gauss_src_file);
        let hyst = Program::builder().src_file(hyst_src_file);
        let nms = Program::builder().src_file(non_max_supp_src_file);
        let sobel = Program::builder().src_file(sobel_src_file);

        ImgPrc{ 
            gauss: gauss,
            hysteresis: hyst,
            non_max_supp: nms,
            sobel: sobel,
            workgroup_size:workgroup_size,
         }
    }

// dim: width, height
    pub fn process_image(self, data: Vec<u8>, dim: (u32, u32)) -> Vec<u8> {
        let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
        let device = context.devices()[0];

       let kdim = dim.0.checked_mul(dim.1).unwrap().checked_mul(mem::size_of::<u8>() as u32).unwrap();

       let gauss = ProQue::builder()
            .context(context.clone())
            .device(device)
            .prog_bldr(self.gauss)
            .dims(kdim)
            .build();

        let hyst = ProQue::builder()
            .context(context.clone())
            .device(device)
            .prog_bldr(self.hysteresis)
            .dims(kdim)
            .build();

        let nms = ProQue::builder()
            .context(context.clone())
            .device(device)
            .prog_bldr(self.non_max_supp)
            .dims(kdim)
            .build();

        let sobel = ProQue::builder()
            .context(context.clone())
            .device(device)
            .prog_bldr(self.sobel)
            .dims(kdim)
            .build();

        let pqs = ProcessingProQues {
            gauss_pq: gauss.unwrap(),
            hyst_pq: hyst.unwrap(),
            nms_pq: nms.unwrap(),
            sobel_pq: sobel.unwrap(),
            lws: self.workgroup_size,
        };

        Canny::new(&pqs, data, dim).execute_edge_detection()
    }


}

struct ProcessingProQues {
    gauss_pq: ProQue,
    hyst_pq: ProQue,
    nms_pq: ProQue,
    sobel_pq: ProQue,
    lws: u32,
}

struct Canny<'a>{
    buffers: Vec<Buffer<u8>>,
    theta_buffer: Buffer<u8>,
    buffer_index: usize,
    proques: &'a ProcessingProQues,
    dim: (u32, u32)
}

impl<'a> Canny<'a> {
    // add code here

    pub fn new(pqs: &'a ProcessingProQues, data: Vec<u8>, dim: (u32, u32)) -> Canny<'a> {
        let next_buffer = Buffer::builder()
            .queue(pqs.gauss_pq.queue().clone())
            .flags(core::MEM_READ_WRITE | core::MEM_ALLOC_HOST_PTR | core::MEM_COPY_HOST_PTR)
            .dims(pqs.gauss_pq.dims().clone())
            .host_data(&data)
            .build().unwrap();
        
        let prev_buffer = Buffer::builder()
            .queue(pqs.gauss_pq.queue().clone())
            .flags(core::MEM_READ_WRITE | core::MEM_ALLOC_HOST_PTR)
            .dims(pqs.gauss_pq.dims().clone())
            .build().unwrap();

        let thetaBuffer = Buffer::builder()
            .queue(pqs.gauss_pq.queue().clone())
            .flags(core::MEM_READ_WRITE | core::MEM_ALLOC_HOST_PTR)
            .dims(pqs.gauss_pq.dims().clone())
            .build().unwrap();

        let buffers = vec![prev_buffer, next_buffer];

        Canny { buffers: buffers, theta_buffer: thetaBuffer, buffer_index: 0, proques: pqs, dim: dim }
    }

    fn next_buffer(&self) -> &Buffer<u8> {
        &self.buffers[self.buffer_index]
    }

    fn prev_buffer(&self) -> &Buffer<u8> {
        &self.buffers[if (self.buffer_index - 1) < 0 { 1 } else { 0 }]
    }

    fn advance_buffer(&mut self) -> &Self {
        self.buffer_index = (self.buffer_index + 1) % 2;

        self
    }

    fn execute_gaussian(&mut self) {
        let kernel = self.proques.gauss_pq.create_kernel("gaussian_kernel").unwrap()
            .gws(self.proques.gauss_pq.dims().clone())
            .lws(self.proques.lws)
            .arg_buf_named("data", Some(self.prev_buffer()))
            .arg_buf_named("out", Some(self.next_buffer()))
            .arg_scl(self.dim.0)
            .arg_scl(self.dim.1);

        kernel.enq().unwrap();
    }

    pub fn execute_edge_detection(&mut self) -> Vec<u8> {
        self.advance_buffer();

        self.execute_gaussian();

        let mut res = vec![0u8; (self.dim.0 * self.dim.1) as usize]; //pretty unsafe

        println!("Result size: {}", res.len());
        self.next_buffer().read(&mut res).enq().unwrap();

        res
    }
}