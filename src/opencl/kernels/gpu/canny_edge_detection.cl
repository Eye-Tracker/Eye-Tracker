__constant float gauss[3][3] = { {0.0625, 0.125, 0.0625},
                                {0.1250, 0.250, 0.1250},
                                {0.0625, 0.125, 0.0625} };

#define BLOCK_DIM 16

// Gaussian Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
__kernel void gaussian_kernel(__global uchar *data,
                              __global uchar *out,
                                       uint width,
                                       uint height) {
    int sum = 0;
    size_t xIndex = get_global_id(0);
    size_t yIndex = get_global_id(1);

    if((xIndex < width - 1) && (xIndex > 0) &&
        (yIndex < height - 1) && (yIndex > 0)) {
        for (int x = 0; x < 3; x++) {
            size_t curX = xIndex - x - 1;
            for (int y = 0; y < 3; y++) {
                size_t curY = yIndex - y - 1;
                sum += gauss[y][x] * data[curY * width + curX];
            }
        }

        out[yIndex * width + xIndex] = min(255,max(0,sum));
    }

    return;
}

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

// Non-maximum Supression Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
// theta: angle input
__kernel void non_max_suppression_kernel(__global uchar *data,
                                        __global uchar *out,
                                        __global uchar *theta,
                                                uint width,
                                                uint height) {
    size_t xIndex = get_global_id(0);
    size_t yIndex = get_global_id(1);
    
    size_t pos = yIndex * width + xIndex;

    uchar my_magnitude = data[pos];

    switch (theta[pos]) {
        // angle of 0 degrees => edge that is North/South
        // neighbors East and West
        case 0:
            // neighbor has larger magnitude
            if (my_magnitude <= data[yIndex * width + xIndex + 1] || // east
                my_magnitude <= data[yIndex * width + xIndex - 1]) {  // west
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                
        // angle of 45 degrees => edge that is NW/SE
        // neighbors NE and SW
        case 45:
            // neighbor has larger magnitude
            if (my_magnitude <= data[(yIndex - 1) * width + xIndex + 1] || // north east
                my_magnitude <= data[(yIndex + 1) * width + xIndex - 1]) {  // south west
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                    
        // angle of 90 degrees => edge that is E/W
        // neighbors North and South
        case 90:
            // neighbor has larger magnitude
            if (my_magnitude <= data[(yIndex - 1) * width + xIndex] || // north
                my_magnitude <= data[(yIndex + 1) * width + xIndex + 1]) { // south
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                    
        // angle of 135 degrees => edge that is NE/SW
        // neighbors NW and SE
        case 135:
            // neighbor has larger magnitude
            if (my_magnitude <= data[(yIndex - 1) * width + xIndex - 1] || // north west
                my_magnitude <= data[(yIndex + 1) * width + xIndex + 1]) { // south east
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                    
        default:
            out[pos] = my_magnitude;
            break;
    } 
}

// Hysteresis Threshold Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
__kernel void hysteresis_kernel(__global uchar *data,
                                __global uchar *out,
                                        float lowThresh,
                                        float highThresh,
                                        uint rows,
                                        uint cols)
{
	// These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

    const uchar EDGE = 255;

    uchar magnitude = data[pos];
    
    if (magnitude >= highThresh)
        out[pos] = EDGE;
    else if (magnitude <= lowThresh)
        out[pos] = 0;
    else
    {
        float med = (highThresh + lowThresh)/2;
        
        if (magnitude >= med)
            out[pos] = EDGE;
        else
            out[pos] = 0;
    }
}