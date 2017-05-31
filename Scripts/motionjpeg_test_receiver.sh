gst-launch-1.0 udpsrc port=5200 ! rtpjpegdepay ! jpegdec ! videoconvert ! ximagesink sync=false -v
