#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char **argv)
{
    // init videocapture
    // VideoCapture cap("tcp://192.168.1.185:5001/", 1900); // 1900 for FFMPEG
    VideoCapture cap;
    do
    {
        cap.open("udp://192.168.1.185:5001?overrun_nonfatal=1&fifo_size=50000000", 1900); // 1900 for FFMPEG
        // cap.open(0);
        cap.set(CAP_PROP_FPS, 120);
        printf("Trying to Open Connection...\n");
    } while(!cap.isOpened());

    // main loop
    double t_start, t_end, average_frame_time;
    double sum_frame_time = 0.0;
    unsigned long frame_count = 0;
    Mat frame;
    while(1)
    {
        // stop time
        t_start = (double) getTickCount();

        // retrieve frame
        cap >> frame;
        
        // stop time (pure)
        t_end = (double(getTickCount()) - t_start)/getTickFrequency();
        
        // show frame
        imshow("Webcam", frame);

        // exit on ESC
        if(waitKey(1) == 27) break;

        // calculate average frame time and print stats
        frame_count++;
        sum_frame_time += t_end;
        if(frame_count == 0) // reset on overflow
        {
            frame_count = 1;
            sum_frame_time = t_end;
        }
        average_frame_time = sum_frame_time/frame_count;
        printf("\033[s"); // store shell-cursor position
        printf("ftime: %f | FPS: %f\n", average_frame_time, 1.0/average_frame_time);
        printf("\033[u"); // restore shell-cursor position
    }
    // move cursor to next line that stats will not get overwritten
    printf("\033[1B");

    return EXIT_SUCCESS;
}