// rgb to grayscale converter
// data: image input (8Bit 3Channel)
// out: image output (8Bit 1Channel)
__kernel void rgb_to_luma_kernel(__global uchar *data,
                                 __global uchar *out,
                                       uint width,
                                       uint height) {
    size_t xIndex = get_global_id(0);
    size_t yIndex = get_global_id(1);

    size_t rgbPos = yIndex * (width * 3) + xIndex * 3;

    if((xIndex < width) && (xIndex > 0) &&
        (yIndex < height) && (yIndex > 0)) {
    
        uchar r = data[rgbPos];
        uchar g = data[rgbPos + 1];
        uchar b = data[rgbPos + 2];
        
        out[yIndex * width + xIndex] = .299f * r + .587f * g + .114f * b;
    }

    return;
}