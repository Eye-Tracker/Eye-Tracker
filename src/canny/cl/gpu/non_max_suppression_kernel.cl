// Non-maximum Supression Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
// theta: angle input
__kernel void non_max_suppression_kernel(__global uchar *data,
                                        __global uchar *out,
                                        __global uchar *theta,
                                                size_t rows,
                                                size_t cols) {
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;
    
    size_t pos = g_row * cols + g_col;
    
    __local int l_data[18][18];

    // copy to l_data
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

    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    else if (l_col == 16)
        l_data[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    uchar my_magnitude = l_data[l_row][l_col];

    switch (theta[pos]) {
        // angle of 0 degrees => edge that is North/South
        // neighbors East and West
        case 0:
            // neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row][l_col+1] || // east
                my_magnitude <= l_data[l_row][l_col-1]) {  // west
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                
        // angle of 45 degrees => edge that is NW/SE
        // neighbors NE and SW
        case 45:
            // neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col+1] || // north east
                my_magnitude <= l_data[l_row+1][l_col-1]) {  // south west
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                    
        // angle of 90 degrees => edge that is E/W
        // neighbors North and South
        case 90:
            // neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col] || // north
                my_magnitude <= l_data[l_row+1][l_col]) { // south
                out[pos] = 0;
            } else {
                out[pos] = my_magnitude;
            }
            break;
                    
        // angle of 135 degrees => edge that is NE/SW
        // neighbors NW and SE
        case 135:
            // neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col-1] || // north west
                my_magnitude <= l_data[l_row+1][l_col+1]) { // south east
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