gst-launch-1.0 udpsrc port=5200 ! tsparse ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! ximagesink sync=false -v