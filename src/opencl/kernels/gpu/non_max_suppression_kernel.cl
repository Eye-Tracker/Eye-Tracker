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