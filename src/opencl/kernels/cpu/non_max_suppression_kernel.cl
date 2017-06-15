// Non-maximum Supression Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
// theta: angle input
__kernel void non_max_suppression_kernel(__global uchar *data,
                                        __global uchar *out,
                                        __global uchar *theta,
                                                uint rows,
                                                uint cols) {
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);

    // Used for matrix addressing
    const size_t POS = row * cols + col;
    const size_t N = (row - 1) * cols + col;
    const size_t NE = (row - 1) * cols + (col + 1);
    const size_t E = row * cols + (col + 1);
    const size_t SE = (row + 1) * cols + (col + 1);
    const size_t S = (row + 1) * cols + col;
    const size_t SW = (row + 1) * cols + (col - 1);
    const size_t W = row * cols + (col - 1);
    const size_t NW = (row - 1) * cols + (col - 1);

    switch (theta[POS]) {
        // angle of 0 degrees => edge that is North/South
        // neighbors East and West
        case 0:
            // neighbor has larger magnitude
            if (data[POS] <= data[E] || data[POS] <= data[W]) {
                out[POS] = 0;
            } else {
                out[POS] = data[POS];
            }
            break;
                
        // angle of 45 degrees => edge that is NW/SE
        // neighbors NE and SW
        case 45:
            // neighbor has larger magnitude
            if (data[POS] <= data[NE] || data[POS] <= data[SW]) {
                out[POS] = 0;
            } else {
                out[POS] = data[POS];
            }
            break;
                    
        // angle of 90 degrees => edge that is E/W
        // neighbors North and South
        case 90:
            // neighbor has larger magnitude
            if (data[POS] <= data[N] || data[POS] <= data[S]) {
                out[POS] = 0;
            } else {
                out[POS] = data[POS];
            }
            break;
                    
        // angle of 135 degrees => edge that is NE/SW
        // neighbors NW and SE
        case 135:
            // neighbor has larger magnitude
            if (data[POS] <= data[NW] || data[POS] <= data[SE]) {
                out[POS] = 0;
            } else {
                out[POS] = data[POS];
            }
            break;
                    
        default:
            out[POS] = data[POS];
            break;
    }
}