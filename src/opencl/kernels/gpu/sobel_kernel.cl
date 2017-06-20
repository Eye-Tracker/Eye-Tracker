// convolution kernels
__constant int sobx[3][3] = { {-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1} };

__constant int soby[3][3] = { {-1,-2,-1},
                              { 0, 0, 0},
                              { 1, 2, 1} };

// Sobel kernel. Apply sobx and soby separately, then find sqrt of squares.
// data:  image input  (8Bit 1Channel)
// out:   image output (8Bit 1Channel)
// theta: angle output
__kernel void sobel_kernel(__global uchar *data,
                           __global uchar *out,
                           __global uchar *theta,
                                    uint width,
                                    uint height)
{
    // collect sums separately
    const float PI = 3.14159265;
    size_t xIndex = get_global_id(0);
    size_t yIndex = get_global_id(1);

    float sumx = 0, sumy = 0, angle = 0;
    if((xIndex < width - 1) && (xIndex > 0) &&
        (yIndex < height - 1) && (yIndex > 0)) {
        for (int x = 0; x < 3; x++) {
            size_t curX = xIndex - x - 1;
            for (int y = 0; y < 3; y++) {
                size_t curY = yIndex - y - 1;
                sumx += sobx[x][y] * data[curY * width + curX];
                sumy += soby[x][y] * data[curY * width + curX];
            }
        }
    }

    // Output is now square root of their square
    out[yIndex * width + xIndex] = min(255,max(0, (int)hypot(sumx,sumy) ));

    // direction angle theta in radians
    angle = atan2(sumy,sumx);

    // if angle is negative, shift the range to (0, 2PI)
    if (angle < 0) {
        angle = fmod((angle + 2*PI),(2*PI));
    }

    // Round the angle to one of four possibilities
    theta[yIndex * width + xIndex] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
}