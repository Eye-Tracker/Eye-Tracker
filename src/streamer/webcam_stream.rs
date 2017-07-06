use streamer::Stream;
use camera_capture;

#[derive(Clone)]
pub struct WebcamStream{
    dim: (u32, u32)
}

impl Stream for WebcamStream {
    fn setup() -> WebcamStream {
        WebcamStream{ dim: (1280, 720) }
    }

    fn fetch_images(&self) -> Box<Iterator<Item = Vec<u8>>> {
        let cam = camera_capture::create(0).unwrap()
                                    .fps(30.0)
                                    .unwrap()
                                    .resolution(self.dim.0, self.dim.1)
                                    .unwrap()
                                    .start()
                                    .unwrap();

        Box::new(cam.map(|img| img.into_raw()))
    }

    fn get_resolution(&self) -> (u32, u32) {
        self.dim
    }
}