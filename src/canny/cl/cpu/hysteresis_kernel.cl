// Hysteresis Threshold Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
__kernel void hysteresis_kernel(__global uchar *data,
                                __global uchar *out,
                                        size_t rows,
                                        size_t cols) {
	// Set the thresholds
	float lowThresh = 10;
	float highThresh = 70;

	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

    const size_t EDGE = 255;
    
    if (data[pos] >= highThresh)
        out[pos] = EDGE;
    else if (data[pos] <= lowThresh)
        out[pos] = 0;
    else {
        float med = (highThresh + lowThresh)/2;
        
        if (data[pos] >= med)
            out[pos] = EDGE;
        else
            out[pos] = 0;
    }
}