//Set kernel path /cl  /cpu or /gpu

//Load Kernels

//Pick device

//16 threads for gpu. 1 for cpu
extern crate ocl;
extern crate find_folder;

use self::ocl::{Device, Context, Queue, Program};
use self::find_folder::Search;

pub struct ImgPrc {
    gauss_prog: Program,
    hysteresis_prog: Program,
    non_max_supp_prog: Program,
    sobel_prog: Program,
}

impl ImgPrc {
    // add code here

    pub fn new(gpu: bool) -> Self {
        let cpu_or_gpu = if gpu { "gpu/" } else { "cpu/" };
        let path = Search::ParentsThenKids(3, 3).for_folder("kernels").unwrap();
        let gauss_src_file = path.join(cpu_or_gpu.to_string() + "gaussian_kernel.cl");
        let hyst_src_file = path.join(cpu_or_gpu.to_string() + "hysteresis_kernel.cl");
        let non_max_supp_src_file = path.join(cpu_or_gpu.to_string() + "non_max_suppression_kernel.cl");
        let sobel_src_file = path.join(cpu_or_gpu.to_string() + "sobel_kernel.cl");

        let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
        let device = context.devices()[0];

        let gauss = Program::builder().src_file(gauss_src_file).devices(device).build(&context).unwrap();
        let hyst = Program::builder().src_file(hyst_src_file).devices(device).build(&context).unwrap();
        let nms = Program::builder().src_file(non_max_supp_src_file).devices(device).build(&context).unwrap();
        let sobel = Program::builder().src_file(sobel_src_file).devices(device).build(&context).unwrap();

        ImgPrc{ 
            gauss_prog: gauss,
            hysteresis_prog: hyst,
            non_max_supp_prog: nms,
            sobel_prog: sobel,
         }
    }
}