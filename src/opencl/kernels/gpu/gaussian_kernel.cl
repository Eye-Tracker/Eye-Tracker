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