use ocl::{Device, Context, Queue};
use ocl::flags::DeviceType;
use find_folder::Search;

use std::path::PathBuf;
use opencl::canny_edge_detection::Canny;
//use ellipse_detect;

pub struct ImgProcessor {
    context: Context,
    queue: Queue,
    path: PathBuf,
    dim: (u32, u32),
}

pub fn setup(gpu: bool, dim: (u32, u32)) -> ImgProcessor {
    let dtype = if gpu { DeviceType::new().gpu() } else { DeviceType::new().cpu() };

    let mut path = Search::ParentsThenKids(3, 3).for_folder("kernels").unwrap();
    if gpu {
        path.push("gpu");
    } else {
        path.push("cpu");
    }

    let context = Context::builder().devices(Device::specifier().type_flags(dtype).first()).build().unwrap();
    let device = context.devices()[0];

    println!("Running on: {} - {}", device.vendor(), device.name());

    let queue = Queue::new(&context, device, None).unwrap();

    ImgProcessor { context: context, queue: queue, path: path, dim: dim }
}

impl ImgProcessor {
    pub fn setup_canny_edge_detection(&self) -> Canny {
        Canny::new(self.path.clone(), &self.context, self.queue.clone(), self.dim)
    }
}