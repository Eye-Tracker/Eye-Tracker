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
                                    size_t rows,
                                    size_t cols)
{
    // collect sums separately
    const float PI = 3.14159265;
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;
    
    size_t pos = g_row * cols + g_col;
    
    __local int l_data[18][18];

    // copy to local
    l_data[l_row][l_col] = data[pos];

    if (l_row == 1) {                           // top most row
        l_data[0][l_col] = data[pos-cols];
        if (l_col == 1)                         // top left
            l_data[0][0] = data[pos-cols-1];

        else if (l_col == 16)                   // top right
            l_data[0][17] = data[pos-cols+1];
    } else if (l_row == 16) {                   // bottom most row
        l_data[17][l_col] = data[pos+cols];
        if (l_col == 1)                         // bottom left
            l_data[17][0] = data[pos+cols-1];
        
        else if (l_col == 16)                   // bottom right
            l_data[17][17] = data[pos+cols+1];
    }

    if (l_col == 1)                             // left
        l_data[l_row][0] = data[pos-1];
    else if (l_col == 16)                       // right
        l_data[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sumx = 0, sumy = 0, angle = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            sumx += sobx[i][j] * l_data[i+l_row-1][j+l_col-1];
            sumy += soby[i][j] * l_data[i+l_row-1][j+l_col-1];
        }
    }

    // Output is now square root of their square
    out[pos] = min(255,max(0, (int)hypot(sumx,sumy) ));

    // direction angle theta in radians
    angle = atan2(sumy,sumx);

    // if angle is negative, shift the range to (0, 2PI)
    if (angle < 0) {
        angle = fmod((angle + 2*PI),(2*PI));
    }

    // Round the angle to one of four possibilities
    theta[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
}