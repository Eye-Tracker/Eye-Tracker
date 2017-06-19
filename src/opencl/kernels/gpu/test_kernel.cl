__kernel void test_kernel(__global uchar *data,
                              __global uchar *out,
                                       uint rows,
                                       uint cols) {
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);

    size_t pos = g_row * cols + g_col;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    out[pos] = pos % 255;
}