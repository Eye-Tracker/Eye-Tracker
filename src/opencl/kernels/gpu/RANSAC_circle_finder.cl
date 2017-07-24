__kernel void ransac_kernel(__global int* const consensus_set_x,
                            __global int* const consensus_set_y,
                            __global int* const consensus_size, //Len num_contours
                            __global int2* center, //Len contours
                            __global float* radius, //Len contours
                            __global int3* randoms, //Len num_contours * num_iterations * 10
                            __local int* votes, //Len num_iterations
                            __local int2* iter_center, //Len num_iterations
                            __local float* iter_radius, //Len num_iterations
                            __private float max_point_seperation,
                            __private float min_point_seperation,
                            __private float colinear_tolerance,
                            __private float radius_tolerance,
                            __private float point_threshold,
                            __private int num_iterations,
                            __private int max_width) {
    int contour_id = get_group_id(0);

    int cons_size = consensus_size[contour_id];
    int start_read = contour_id * max_width;

    iter_center[get_local_id(0)] = (int2)(0);
    iter_radius[get_local_id(0)] = 0;
    if(get_local_id(0) == 0) {
        center[contour_id * 2] = (int2)(0);
        radius[contour_id] = 0;
    }

    int iA, iB, iC;
    float AB, BC, CA;
    float m_AB, m_BC, b_AB;
    float x_mp_AB, y_mp_AB, x_mp_BC, y_mp_BC;
    float m_pb_AB, m_pb_BC, b_pb_AB, b_pb_BC;
    int MAX_SUB_ITER = 10;
    int sub_iter = 0;
    for(; sub_iter < MAX_SUB_ITER;) {
        iA = randoms[get_global_id(0) * 10 + sub_iter].x;
        iB = randoms[get_global_id(0) * 10 + sub_iter].y;
        iC = randoms[get_global_id(0) * 10 + sub_iter].z;
        iA = iA % cons_size;
        iB = iB % cons_size;
        iC = iC % cons_size;

        AB = fast_length((float3)(
                consensus_set_x[start_read + iA] - consensus_set_x[start_read + iB],
                consensus_set_y[start_read + iA] - consensus_set_y[start_read + iB], 0.0));
        BC = fast_length((float3)(
                consensus_set_x[start_read + iB] - consensus_set_x[start_read + iC],
                consensus_set_y[start_read + iB] - consensus_set_y[start_read + iC], 0.0));
        CA = fast_length((float3)(
                consensus_set_x[start_read + iC] - consensus_set_x[start_read + iA],
                consensus_set_y[start_read + iC] - consensus_set_y[start_read + iA], 0.0));

        if (AB < min_point_seperation || BC < min_point_seperation ||
            CA < min_point_seperation ||
            AB > max_point_seperation || BC > max_point_seperation ||
            CA > max_point_seperation) {
            sub_iter++;
        } else {
            break;
        }
    }
    if (sub_iter >= MAX_SUB_ITER) {
        return;
    }

    m_AB = (consensus_set_y[start_read + iA] - consensus_set_y[start_read + iB]) /
            (consensus_set_x[start_read + iA] - consensus_set_x[start_read + iB] + 0.001);
    m_BC = (consensus_set_y[start_read + iB] - consensus_set_y[start_read + iC]) /
            (consensus_set_x[start_read + iB] - consensus_set_x[start_read + iC] + 0.001);
    b_AB = (consensus_set_y[start_read + iB] - m_AB * consensus_set_x[start_read + iB]);

    if(abs( (int)(consensus_set_y[start_read + iC] - ((m_AB * consensus_set_x[start_read + iC]) + b_AB)) ) < colinear_tolerance) {
        return;
    }

    x_mp_AB = (consensus_set_x[start_read + iA] + consensus_set_x[start_read + iB])/2.0;
    y_mp_AB = (consensus_set_y[start_read + iA] + consensus_set_y[start_read + iB])/2.0;
    x_mp_BC = (consensus_set_x[start_read + iB] + consensus_set_x[start_read + iC])/2.0;
	y_mp_BC = (consensus_set_y[start_read + iB] + consensus_set_y[start_read + iC])/2.0;

    m_pb_AB = -1/m_AB;
    m_pb_BC = -1/m_BC;
    b_pb_AB = y_mp_AB - m_pb_AB * x_mp_AB;
    b_pb_BC = y_mp_BC - m_pb_BC * x_mp_BC;

    int cx = (b_pb_AB - b_pb_BC)/(m_pb_BC - m_pb_AB + 0.0001);
    int cy = m_pb_AB * cx + b_pb_AB;
    float cur_radius = fast_length((float3)(
        cx - consensus_set_x[start_read + iA],
        cx - consensus_set_y[start_read + iA], 0.0
    ));

    iter_center[get_local_id(0)] = (int2)(cx, cy);
    iter_radius[get_local_id(0)] = cur_radius;

    if (cx < 0 || cy < 0) {
        printf("\nOCL:\nIn Thread %d %d %d %f\nContour: %d\nRands: %d, %d, %d\nConsensus Size: %d\nFinal Indices: %d, %d, %d\nAB, BC, CA: %f %f %f\nConsA: %d %d\nConsB: %d %d\nConsC: %d %d\nm_AB, m_BC, b_AB: %f, %f, %f\nx_mp_AB, y_mp_AB, x_mp_BC, y_mp_BC: %f, %f, %f, %f\nm_pb_AB, m_pb_BC: %f, %f\nb_pb_AB, b_pb_BC: %f, %f\n", 
        	get_local_id(0), cx, cy, cur_radius, 
        	contour_id,
        	iA, iB, iC,
        	AB, BC, CA,
        	cons_size,
        	start_read + iA, start_read + iB, start_read + iC,
        	consensus_set_x[start_read + iA], consensus_set_y[start_read + iA],
        	consensus_set_x[start_read + iB], consensus_set_y[start_read + iB],
        	consensus_set_x[start_read + iC], consensus_set_y[start_read + iC],
        	m_AB, m_BC, b_AB,
        	x_mp_AB, y_mp_AB, x_mp_BC, y_mp_BC,
        	m_pb_AB, m_pb_BC,
        	b_pb_AB, b_pb_BC);
    }

    votes[get_local_id(0)] = 0;
    for(int i = 0; i < consensus_size[contour_id]; i++) {
        if (
        	fast_length((float3)(consensus_set_y[start_read + i] - cy,
                       	consensus_set_x[start_read + i] - cx, 0.0)) - cur_radius
             < radius_tolerance) {
            votes[get_local_id(0)]++;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if(get_local_id(0)) {
        int max_votes = 0;
        int max_iter = -1;
        for (int i = 0; i < num_iterations; i++) {
            if (votes[i] > max_votes) {
                max_votes = votes[i];
                max_iter = i;
            }
        }

        
        center[contour_id] = iter_center[max_iter];
        radius[contour_id] = iter_radius[max_iter];
    }
}