gst-launch-1.0 v4l2src ! video/x-raw,width=640,height=480 ! x264enc ! h264parse ! rtph264pay ! udpsink host=$1 port=5200
