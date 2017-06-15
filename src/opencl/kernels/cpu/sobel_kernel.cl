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
                                    uint rows,
                                    uint cols) {
    // collect sums separately
    const float PI = 3.14159265;
    float sumx = 0, sumy = 0, angle = 0;
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);
    size_t pos = row * cols + col;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sumx += sobx[i][j] *
                    data[ (i+row+-1)*cols + (j+col+-1) ];
            sumy += soby[i][j] *
                    data[ (i+row+-1)*cols + (j+col+-1) ];
        }
    }

    // Output is now square root of their square
    out[pos] = min(255,max(0, (int)hypot(sumx,sumy) ));

    // direction angle theta in radians
    angle = atan2(sumy,sumx);

    
    if (angle < 0)
        angle = fmod((angle + 2*PI),(2*PI));

    if (angle <= PI/8) {
        theta[pos] = 0;
    } else if (angle <= 3*PI/8) {
        theta[pos] = 45;
    } else if (angle <= 5*PI/8) {
        theta[pos] = 90;
    } else if (angle <= 7*PI/8) {
        theta[pos] = 135;
    } else if (angle <= 9*PI/8) {
        theta[pos] = 0;
    } else if (angle <= 11*PI/8) {
        theta[pos] = 45;
    } else if (angle <= 13*PI/8) {
        theta[pos] = 90;
    } else if (angle <= 15*PI/8) {
        theta[pos] = 135;
    } else { // (angle <= 16*PI/8) 
        theta[pos] = 0;
    }
}