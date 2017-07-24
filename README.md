# Eye-Tracker-GPU Implementation

Implementation of some eye tracking algorithms on the GPU. Everything is implemented in Rust [https://www.rust-lang.org/en-US/](link) and OpenCL. The project currently builds with the nightly version of Rust on Windows with the MSVC compiler (rustc 1.19.0-nightly). 

For OpenCL the AMD APP SDK needs to be installed to the default location.

This project should be able to run on Linux too. On OSX webcam capturing isn't supported. If you still want to try it you can run it with:

```
cargo run --release --no-default-features --features "use_dummy_streamer"
```

# Project overview

- main.rs: Spawns GUI, receives images and send them to processing
- contour_detection/: Part of the project which executes contour detection
- gui/: Handles the configuration UI
- image_loader/: Small helper to load and save images. Not really used currently
- opencl/: Part of the project which handles the Canny Edge detection and the RANSAC Ellipse fitting
- streamer/: Part of the project which provides different image capture methods 

# Used algorithms:

- **Contour detection**: S. Suzuki and K. Abe - Topological Structural Analysis of Digitized Binary Image by Border Following 
- **Canny Edge Detection**: John Canny - A Computational Approach to Edge Detection
- **RANSAC Ellipse Estimation**: Based on [The Insight Journal - 2010 July-December. ](http://www.insight-journal.org/browse/publication/769) and it's implementation in [cpp](https://github.com/midas-journal/midas-journal-769)