// grayscale to rgba converter
// data: image input (8Bit 1Channel)
// out: image output (8Bit 4Channel)
__kernel void luma_to_rgba_kernel(__global uchar *data,
                                 __global uchar *out,
                                       uint width,
                                       uint height) {
    size_t xIndex = get_global_id(0);
    size_t yIndex = get_global_id(1);

    size_t rgbaPos = yIndex * (width * 4) + xIndex * 4;

    if((xIndex < width) && (xIndex > 0) &&
        (yIndex < height) && (yIndex > 0)) {
        
        out[rgbaPos] = data[yIndex * width + xIndex];
        out[rgbaPos + 1] = data[yIndex * width + xIndex];
        out[rgbaPos + 2] = data[yIndex * width + xIndex];
        out[rgbaPos + 3] = 255;
    }

    return;
}