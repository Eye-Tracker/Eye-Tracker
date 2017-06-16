// Hysteresis Threshold Kernel
// data: image input (8Bit 1Channel)
// out: image output (8Bit 1Channel)
__kernel void hysteresis_kernel(__global uchar *data,
                                __global uchar *out,
                                        uint rows,
                                        uint cols)
{
	// Establish our high and low thresholds as floats
	float lowThresh = 10;
	float highThresh = 70;

	// These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

    if (row == 0 || row == rows -1 || col == 0 || col == cols - 1) {
        return;
    }

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