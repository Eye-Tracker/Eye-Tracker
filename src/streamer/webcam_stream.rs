use streamer::Stream;
use camera_capture;
use image;

#[derive(Clone)]
pub struct WebcamStream{
    dim: (u32, u32)
}

impl Stream for WebcamStream {
    fn setup() -> WebcamStream {
        WebcamStream{ dim: (640, 480) }
    }

    fn fetch_images(&self) -> Box<Iterator<Item = image::RgbImage>> {
        let cam = camera_capture::create(0).unwrap()
                                    .fps(30.0)
                                    .unwrap()
                                    .resolution(self.dim.0, self.dim.1)
                                    .unwrap()
                                    .start()
                                    .unwrap();

        Box::new(cam.map(|img| img))
    }

    fn get_resolution(&self) -> (u32, u32) {
        self.dim
    }
}