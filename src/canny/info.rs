extern crate ocl;

use self::ocl::{Platform, Device, Context, Queue, Buffer, Image, Sampler, Program, Kernel, Event, EventList};
use self::ocl::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ImageInfo,
    SamplerInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo,
    EventInfo, ProfilingInfo};
use self::ocl::util;

const DIMS: [usize; 3] = [1024, 64, 16];
const INFO_FORMAT_MULTILINE: bool = true;

static SRC: &'static str = r#"
    __kernel void multiply(float coeff, __global float* buffer) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

pub fn print_info() {
    let platforms = Platform::list();
    for (plat_idx, &platform) in platforms.iter().enumerate() {
        print_platform(plat_idx, platform);
    }
}

fn print_platform(plat_idx: usize, platform: Platform) {
    for (device_idx, &device) in Device::list_all(&platform).unwrap().iter().enumerate() {
        print_platform_device(plat_idx, platform, device_idx, device);
    }
}

fn print_platform_device(plat_idx: usize, platform: Platform, device_idx: usize, device: Device) {
    let device_version = device.version().unwrap();

    let context = Context::builder().platform(platform).devices(device).build().unwrap();
    let program = Program::builder()
        .devices(device)
        .src(SRC)
        .build(&context).unwrap();
    let queue = Queue::new(&context, device, Some(core::QUEUE_PROFILING_ENABLE)).unwrap();
    let buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .dims(&DIMS)
        .build().unwrap();
    let image = Image::<u8>::builder()
        .dims(&DIMS)
        .queue(queue.clone())
        .build().unwrap();
    let sampler = Sampler::with_defaults(&context).unwrap();
        let kernel = Kernel::new("multiply", &program).unwrap()
        .queue(queue.clone())
        .gws(&DIMS)
        .arg_scl(10.0f32)
        .arg_buf(&buffer);

    let mut event_list = EventList::new();
    kernel.cmd().enew(&mut event_list).enq().unwrap();
    event_list.wait_for().unwrap();

    let mut event = Event::empty();
    buffer.cmd().write(&vec![0.0; DIMS[0]]).enew(&mut event).enq().unwrap();
    event.wait_for().unwrap();

    println!("############### OpenCL Platform-Device Full Info ################");
    print!("\n");

    let (begin, delim, end) = if INFO_FORMAT_MULTILINE {
        ("\n", "\n", "\n")
    } else {
        ("{ ", ", ", " }")
    };

    // ##################################################
    // #################### PLATFORM ####################
    // ##################################################

    println!("Platform [{}]:\n\
            {t}Profile: {}\n\
            {t}Version: {}\n\
            {t}Name: {}\n\
            {t}Vendor: {}\n\
            {t}Extensions: {}\n\
        ",
        plat_idx,
        core::get_platform_info(context.platform().unwrap().unwrap_or(Platform::default()), PlatformInfo::Profile),
        core::get_platform_info(context.platform().unwrap().unwrap_or(Platform::default()), PlatformInfo::Version),
        core::get_platform_info(context.platform().unwrap().unwrap_or(Platform::default()), PlatformInfo::Name),
        core::get_platform_info(context.platform().unwrap().unwrap_or(Platform::default()), PlatformInfo::Vendor),
        core::get_platform_info(context.platform().unwrap().unwrap_or(Platform::default()), PlatformInfo::Extensions),
        t = util::colors::TAB,
    );


    // ##################################################
    // #################### DEVICES #####################
    // ##################################################

    // [FIXME]: Complete this section.
    // [FIXME]: Implement `Display`/`Debug` for all variants of `DeviceInfoResult`.
    // Printing algorithm is highly janky (due to laziness).

    // for device in context.devices().iter() {
    debug_assert!(context.devices().len() == 1);

    // for device_idx in 0..context.devices().len() {
        // let device = context.devices()[device_idx].clone();

        println!("Device [{}]: \n\
                {t}Type: {}\n\
                {t}VendorId: {}\n\
                {t}MaxComputeUnits: {}\n\
                {t}MaxWorkItemDimensions: {}\n\
                {t}MaxWorkGroupSize: {}\n\
                {t}MaxWorkItemSizes: {}\n\
                {t}PreferredVectorWidthChar: {}\n\
                {t}PreferredVectorWidthShort: {}\n\
                {t}PreferredVectorWidthInt: {}\n\
                {t}PreferredVectorWidthLong: {}\n\
                {t}PreferredVectorWidthFloat: {}\n\
                {t}PreferredVectorWidthDouble: {}\n\
                {t}MaxClockFrequency: {}\n\
                {t}AddressBits: {}\n\
                {t}MaxReadImageArgs: {}\n\
                {t}MaxWriteImageArgs: {}\n\
                {t}MaxMemAllocSize: {}\n\
                {t}Image2dMaxWidth: {}\n\
                {t}Image2dMaxHeight: {}\n\
                {t}Image3dMaxWidth: {}\n\
                {t}Image3dMaxHeight: {}\n\
                {t}Image3dMaxDepth: {}\n\
                {t}ImageSupport: {}\n\
                {t}MaxParameterSize: {}\n\
                {t}MaxSamplers: {}\n\
                {t}MemBaseAddrAlign: {}\n\
                {t}MinDataTypeAlignSize: {}\n\
                {t}SingleFpConfig: {}\n\
                {t}GlobalMemCacheType: {}\n\
                {t}GlobalMemCachelineSize: {}\n\
                {t}GlobalMemCacheSize: {}\n\
                {t}GlobalMemSize: {}\n\
                {t}MaxConstantBufferSize: {}\n\
                {t}MaxConstantArgs: {}\n\
                {t}LocalMemType: {}\n\
                {t}LocalMemSize: {}\n\
                {t}ErrorCorrectionSupport: {}\n\
                {t}ProfilingTimerResolution: {}\n\
                {t}EndianLittle: {}\n\
                {t}Available: {}\n\
                {t}CompilerAvailable: {}\n\
                {t}ExecutionCapabilities: {}\n\
                {t}QueueProperties: {}\n\
                {t}Name: {}\n\
                {t}Vendor: {}\n\
                {t}DriverVersion: {}\n\
                {t}Profile: {}\n\
                {t}Version: {}\n\
                {t}Extensions: {}\n\
                {t}Platform: {}\n\
                {t}DoubleFpConfig: {}\n\
                {t}HalfFpConfig: {}\n\
                {t}PreferredVectorWidthHalf: {}\n\
                {t}HostUnifiedMemory: {}\n\
                {t}NativeVectorWidthChar: {}\n\
                {t}NativeVectorWidthShort: {}\n\
                {t}NativeVectorWidthInt: {}\n\
                {t}NativeVectorWidthLong: {}\n\
                {t}NativeVectorWidthFloat: {}\n\
                {t}NativeVectorWidthDouble: {}\n\
                {t}NativeVectorWidthHalf: {}\n\
                {t}OpenclCVersion: {}\n\
                {t}LinkerAvailable: {}\n\
                {t}BuiltInKernels: {}\n\
                {t}ImageMaxBufferSize: {}\n\
                {t}ImageMaxArraySize: {}\n\
                {t}ParentDevice: {}\n\
                {t}PartitionMaxSubDevices: {}\n\
                {t}PartitionProperties: {}\n\
                {t}PartitionAffinityDomain: {}\n\
                {t}PartitionType: {}\n\
                {t}ReferenceCount: {}\n\
                {t}PreferredInteropUserSync: {}\n\
                {t}PrintfBufferSize: {}\n\
                {t}ImagePitchAlignment: {}\n\
                {t}ImageBaseAddressAlignment: {}\n\
            ",
            device_idx,
            core::get_device_info(&device, DeviceInfo::Type),
            core::get_device_info(&device, DeviceInfo::VendorId),
            core::get_device_info(&device, DeviceInfo::MaxComputeUnits),
            core::get_device_info(&device, DeviceInfo::MaxWorkItemDimensions),
            core::get_device_info(&device, DeviceInfo::MaxWorkGroupSize),
            core::get_device_info(&device, DeviceInfo::MaxWorkItemSizes),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthChar),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthShort),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthInt),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthLong),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthFloat),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthDouble),
            core::get_device_info(&device, DeviceInfo::MaxClockFrequency),
            core::get_device_info(&device, DeviceInfo::AddressBits),
            core::get_device_info(&device, DeviceInfo::MaxReadImageArgs),
            core::get_device_info(&device, DeviceInfo::MaxWriteImageArgs),
            core::get_device_info(&device, DeviceInfo::MaxMemAllocSize),
            core::get_device_info(&device, DeviceInfo::Image2dMaxWidth),
            core::get_device_info(&device, DeviceInfo::Image2dMaxHeight),
            core::get_device_info(&device, DeviceInfo::Image3dMaxWidth),
            core::get_device_info(&device, DeviceInfo::Image3dMaxHeight),
            core::get_device_info(&device, DeviceInfo::Image3dMaxDepth),
            core::get_device_info(&device, DeviceInfo::ImageSupport),
            core::get_device_info(&device, DeviceInfo::MaxParameterSize),
            core::get_device_info(&device, DeviceInfo::MaxSamplers),
            core::get_device_info(&device, DeviceInfo::MemBaseAddrAlign),
            core::get_device_info(&device, DeviceInfo::MinDataTypeAlignSize),
            core::get_device_info(&device, DeviceInfo::SingleFpConfig),
            core::get_device_info(&device, DeviceInfo::GlobalMemCacheType),
            core::get_device_info(&device, DeviceInfo::GlobalMemCachelineSize),
            core::get_device_info(&device, DeviceInfo::GlobalMemCacheSize),
            core::get_device_info(&device, DeviceInfo::GlobalMemSize),
            core::get_device_info(&device, DeviceInfo::MaxConstantBufferSize),
            core::get_device_info(&device, DeviceInfo::MaxConstantArgs),
            core::get_device_info(&device, DeviceInfo::LocalMemType),
            core::get_device_info(&device, DeviceInfo::LocalMemSize),
            core::get_device_info(&device, DeviceInfo::ErrorCorrectionSupport),
            core::get_device_info(&device, DeviceInfo::ProfilingTimerResolution),
            core::get_device_info(&device, DeviceInfo::EndianLittle),
            core::get_device_info(&device, DeviceInfo::Available),
            core::get_device_info(&device, DeviceInfo::CompilerAvailable),
            core::get_device_info(&device, DeviceInfo::ExecutionCapabilities),
            core::get_device_info(&device, DeviceInfo::QueueProperties),
            core::get_device_info(&device, DeviceInfo::Name),
            core::get_device_info(&device, DeviceInfo::Vendor),
            core::get_device_info(&device, DeviceInfo::DriverVersion),
            core::get_device_info(&device, DeviceInfo::Profile),
            core::get_device_info(&device, DeviceInfo::Version),
            core::get_device_info(&device, DeviceInfo::Extensions),
            core::get_device_info(&device, DeviceInfo::Platform),
            core::get_device_info(&device, DeviceInfo::DoubleFpConfig),
            core::get_device_info(&device, DeviceInfo::HalfFpConfig),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthHalf),
            core::get_device_info(&device, DeviceInfo::HostUnifiedMemory),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthChar),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthShort),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthInt),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthLong),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthFloat),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthDouble),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthHalf),
            core::get_device_info(&device, DeviceInfo::OpenclCVersion),
            core::get_device_info(&device, DeviceInfo::LinkerAvailable),
            core::get_device_info(&device, DeviceInfo::BuiltInKernels),
            core::get_device_info(&device, DeviceInfo::ImageMaxBufferSize),
            core::get_device_info(&device, DeviceInfo::ImageMaxArraySize),
            core::get_device_info(&device, DeviceInfo::ParentDevice),
            core::get_device_info(&device, DeviceInfo::PartitionMaxSubDevices),
            core::get_device_info(&device, DeviceInfo::PartitionProperties),
            core::get_device_info(&device, DeviceInfo::PartitionAffinityDomain),
            core::get_device_info(&device, DeviceInfo::PartitionType),
            core::get_device_info(&device, DeviceInfo::ReferenceCount),
            core::get_device_info(&device, DeviceInfo::PreferredInteropUserSync),
            core::get_device_info(&device, DeviceInfo::PrintfBufferSize),
            core::get_device_info(&device, DeviceInfo::ImagePitchAlignment),
            core::get_device_info(&device, DeviceInfo::ImageBaseAddressAlignment),
            t = util::colors::TAB,
        );

    // ##################################################
    // #################### CONTEXT #####################
    // ##################################################

    println!("Context:\n\
            {t}Reference Count: {}\n\
            {t}Devices: {}\n\
            {t}Properties: {}\n\
            {t}Device Count: {}\n\
        ",
        core::get_context_info(&context, ContextInfo::ReferenceCount),
        core::get_context_info(&context, ContextInfo::Devices),
        core::get_context_info(&context, ContextInfo::Properties),
        core::get_context_info(&context, ContextInfo::NumDevices),
        t = util::colors::TAB,
    );

    // ##################################################
    // ##################### QUEUE ######################
    // ##################################################


    println!("Command Queue:\n\
            {t}Context: {}\n\
            {t}Device: {}\n\
            {t}ReferenceCount: {}\n\
            {t}Properties: {}\n\
        ",
        core::get_command_queue_info(&queue, CommandQueueInfo::Context),
        core::get_command_queue_info(&queue, CommandQueueInfo::Device),
        core::get_command_queue_info(&queue, CommandQueueInfo::ReferenceCount),
        core::get_command_queue_info(&queue, CommandQueueInfo::Properties),
        t = util::colors::TAB,
    );


    println!("Buffer Memory:\n\
            {t}Type: {}\n\
            {t}Flags: {}\n\
            {t}Size: {}\n\
            {t}HostPtr: {}\n\
            {t}MapCount: {}\n\
            {t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}AssociatedMemobject: {}\n\
            {t}Offset: {}\n\
        ",
        core::get_mem_object_info(&buffer, MemInfo::Type),
        core::get_mem_object_info(&buffer, MemInfo::Flags),
        core::get_mem_object_info(&buffer, MemInfo::Size),
        core::get_mem_object_info(&buffer, MemInfo::HostPtr),
        core::get_mem_object_info(&buffer, MemInfo::MapCount),
        core::get_mem_object_info(&buffer, MemInfo::ReferenceCount),
        core::get_mem_object_info(&buffer, MemInfo::Context),
        core::get_mem_object_info(&buffer, MemInfo::AssociatedMemobject),
        core::get_mem_object_info(&buffer, MemInfo::Offset),
        t = util::colors::TAB,
    );


        println!("Image: {b}\
                {t}ElementSize: {}{d}\
                {t}RowPitch: {}{d}\
                {t}SlicePitch: {}{d}\
                {t}Width: {}{d}\
                {t}Height: {}{d}\
                {t}Depth: {}{d}\
                {t}ArraySize: {}{d}\
                {t}Buffer: {}{d}\
                {t}NumMipLevels: {}{d}\
                {t}NumSamples: {}{e}\
            ",
            core::get_image_info(&image, ImageInfo::ElementSize),
            core::get_image_info(&image, ImageInfo::RowPitch),
            core::get_image_info(&image, ImageInfo::SlicePitch),
            core::get_image_info(&image, ImageInfo::Width),
            core::get_image_info(&image, ImageInfo::Height),
            core::get_image_info(&image, ImageInfo::Depth),
            core::get_image_info(&image, ImageInfo::ArraySize),
            core::get_image_info(&image, ImageInfo::Buffer),
            core::get_image_info(&image, ImageInfo::NumMipLevels),
            core::get_image_info(&image, ImageInfo::NumSamples),
            b = begin,
            d = delim,
            e = end,
            t = util::colors::TAB,
        );

        println!("{t}Image Memory:\n\
                {t}{t}Type: {}\n\
                {t}{t}Flags: {}\n\
                {t}{t}Size: {}\n\
                {t}{t}HostPtr: {}\n\
                {t}{t}MapCount: {}\n\
                {t}{t}ReferenceCount: {}\n\
                {t}{t}Context: {}\n\
                {t}{t}AssociatedMemobject: {}\n\
                {t}{t}Offset: {}\n\
            ",
            core::get_mem_object_info(&buffer, MemInfo::Type),
            core::get_mem_object_info(&buffer, MemInfo::Flags),
            core::get_mem_object_info(&buffer, MemInfo::Size),
            core::get_mem_object_info(&buffer, MemInfo::HostPtr),
            core::get_mem_object_info(&buffer, MemInfo::MapCount),
            core::get_mem_object_info(&buffer, MemInfo::ReferenceCount),
            core::get_mem_object_info(&buffer, MemInfo::Context),
            core::get_mem_object_info(&buffer, MemInfo::AssociatedMemobject),
            core::get_mem_object_info(&buffer, MemInfo::Offset),
            t = util::colors::TAB,
        );

    println!("Sampler:\n\
            {t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}NormalizedCoords: {}\n\
            {t}AddressingMode: {}\n\
            {t}FilterMode: {}\n\
        ",
        core::get_sampler_info(&sampler, SamplerInfo::ReferenceCount),
        core::get_sampler_info(&sampler, SamplerInfo::Context),
        core::get_sampler_info(&sampler, SamplerInfo::NormalizedCoords),
        core::get_sampler_info(&sampler, SamplerInfo::AddressingMode),
        core::get_sampler_info(&sampler, SamplerInfo::FilterMode),
        t = util::colors::TAB,
    );

    println!("Program:\n\
            {t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}NumDevices: {}\n\
            {t}Devices: {}\n\
            {t}Source: {}\n\
            {t}BinarySizes: {}\n\
            {t}Binaries: {}\n\
            {t}NumKernels: {}\n\
            {t}KernelNames: {}\n\
        ",
        core::get_program_info(&program, ProgramInfo::ReferenceCount),
        core::get_program_info(&program, ProgramInfo::Context),
        core::get_program_info(&program, ProgramInfo::NumDevices),
        core::get_program_info(&program, ProgramInfo::Devices),
        core::get_program_info(&program, ProgramInfo::Source),
        core::get_program_info(&program, ProgramInfo::BinarySizes),
        //core::get_program_info(&program, ProgramInfo::Binaries),
        "n/a",
        core::get_program_info(&program, ProgramInfo::NumKernels),
        core::get_program_info(&program, ProgramInfo::KernelNames),
        t = util::colors::TAB,
    );

    println!("Program Build:\n\
            {t}BuildStatus: {}\n\
            {t}BuildOptions: {}\n\
            {t}BuildLog: \n\n{}\n\n\
            {t}BinaryType: {}\n\
        ",
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildStatus),
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildOptions),
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildLog),
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BinaryType),
        t = util::colors::TAB,
    );

    println!("Kernel Info:\n\
            {t}FunctionName: {}\n\
            {t}NumArgs: {}\n\
            {t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}Program: {}\n\
            {t}Attributes: {}\n\
        ",
        core::get_kernel_info(&kernel, KernelInfo::FunctionName),
        core::get_kernel_info(&kernel, KernelInfo::NumArgs),
        core::get_kernel_info(&kernel, KernelInfo::ReferenceCount),
        core::get_kernel_info(&kernel, KernelInfo::Context),
        core::get_kernel_info(&kernel, KernelInfo::Program),
        core::get_kernel_info(&kernel, KernelInfo::Attributes),
        t = util::colors::TAB,
    );

    println!("Kernel Argument [0]:\n\
            {t}AddressQualifier: {}\n\
            {t}AccessQualifier: {}\n\
            {t}TypeName: {}\n\
            {t}TypeQualifier: {}\n\
            {t}Name: {}\n\
        ",
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::AddressQualifier, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::AccessQualifier, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::TypeName, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::TypeQualifier, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::Name, Some(&[device_version])),
        t = util::colors::TAB,
    );

    println!("Kernel Work Group:\n\
            {t}WorkGroupSize: {}\n\
            {t}CompileWorkGroupSize: {}\n\
            {t}LocalMemSize: {}\n\
            {t}PreferredWorkGroupSizeMultiple: {}\n\
            {t}PrivateMemSize: {}\n\
            {t}GlobalWorkSize: {}\n\
        ",
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::WorkGroupSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::CompileWorkGroupSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::LocalMemSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::PrivateMemSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::GlobalWorkSize),
        t = util::colors::TAB,
    );

    println!("Event:\n\
            {t}CommandQueue: {}\n\
            {t}CommandType: {}\n\
            {t}ReferenceCount: {}\n\
            {t}CommandExecutionStatus: {}\n\
            {t}Context: {}\n\
        ",
        core::get_event_info(&event, EventInfo::CommandQueue),
        core::get_event_info(&event, EventInfo::CommandType),
        core::get_event_info(&event, EventInfo::ReferenceCount),
        core::get_event_info(&event, EventInfo::CommandExecutionStatus),
        core::get_event_info(&event, EventInfo::Context),
        t = util::colors::TAB,
    );

    println!("Event Profiling:\n\
            {t}Queued: {}\n\
            {t}Submit: {}\n\
            {t}Start: {}\n\
            {t}End: {}\n\
        ",
        core::get_event_profiling_info(&event, ProfilingInfo::Queued),
        core::get_event_profiling_info(&event, ProfilingInfo::Submit),
        core::get_event_profiling_info(&event, ProfilingInfo::Start),
        core::get_event_profiling_info(&event, ProfilingInfo::End),
        t = util::colors::TAB,
    );

    print!("\n");
}