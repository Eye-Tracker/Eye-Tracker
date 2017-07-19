__kernel void ransac_kernel(__constant int** consensus_set_x,
                            __constant int** consensus_set_y,
                            __global int2* center, //len threads //contours
                            __global float* radius,
                            __constant int3* randoms, //len num_iterations
                            float max_point_seperation,
                            float min_point_seperation,
                            float colinear_tolerance,
                            float radius_tolerance,
                            float point_threshold,
                            int num_iterations,
                            int* consensus_size) { //len threads?
    int contour_id = get_group_id(0);
    __local int votes[num_iterations];
    __local int2 iter_center[num_iterations];
    __local float iter_radius[num_iterations];

    int cons_size = consensus_size[0];

    int global_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    int seed = seed_memory[global_id];

    iter_center[get_local_id(0)] = (int2)(0);
    iter_radius[get_local_id(0)] = 0;
    if(get_local_id(0) == 0) {
        center[contour_id] = (int2)(0);
        radius[contour_id] = 0;
    }

    float iA, iB, iC;
    float AB, BC, CA;
    float m_AB, m_BC, b_AB;
    float x_mp_AB, y_mp_AB, x_mp_BC, y_mp_BC;
    float m_pb_AB, m_pb_BC, b_pb_AB, b_pb_BC;
    int MAX_SUB_ITER = 10;
    for(int sub_iter = 0; sub_iter < MAX_SUB_ITER;) {
        iA = randoms[get_local_id(0)].x;
        iB = randoms[get_local_id(0)].y;
        iC = randoms[get_local_id(0)].z;
        iA %= cons_size;
        iB %= cons_size;
        iC %= cons_size;

        AB = fast_length((int3)(
                consensus_set_x[contour_id][iA] - consensus_set_x[contour_id][iB],
                consensus_set_y[contour_id][iA] - consensus_set_y[contour_id][iB], 0.0));
        BC = fast_length((int3)(
                consensus_set_x[contour_id][iB] - consensus_set_x[contour_id][iC],
                consensus_set_y[contour_id][iB] - consensus_set_y[contour_id][iC], 0.0));
        CA = fast_length((int3)(
                consensus_set_x[contour_id][iC] - consensus_set_x[contour_id][iA],
                consensus_set_y[contour_id][iC] - consensus_set_y[contour_id][iA], 0.0));

        if (AB < min_point_seperation || BC < min_point_seperation ||
            CA < min_point_seperation ||
            AB > max_point_seperation || BC > max_point_seperation ||
            CA > max_point_seperation) {
            sub_iter++
        } else {
            break;
        }
    }
    if (sub_iter >= MAX_SUB_ITER) {
        return;
    }

    m_AB = (consensus_set_y[contour_id][iA] - consensus_set_y[contour_id][iB]) /
            (consensus_set_x[contour_id][iA] - consensus_set_x[contour_id][iB] + 0.001);
    m_BC = (consensus_set_y[contour_id][iB] - consensus_set_y[contour_id][iC]) /
            (consensus_set_x[contour_id][iB] - consensus_set_x[contour_id][iC] + 0.001);
    b_AB = (consensus_set_y[contour_id][iB] - m_AB * consensus_set_x[contour_id][iB]);

    if(abs(consensus_set_y[contour_id][iC] - ((m_AB * consensus_set_x[contour_id][iC]) + b_AB)) < colinear_tolerance) {
        return;
    }

    m_pb_AB = -1/m_AB;
    m_pb_BC = -1/m_BC;
    b_pb_AB = y_mp_AB - m_pb_AB * x_mp_AB;
    p_pb_BC = y_mp_BC - m_pb_BC * x_mp_BC;

    int cx = (b_pb_AB - b_pb_BC)/(m_pb_BC - m_pb_AB + 0.0001);
    int cy = m_pb_AB * cx + b_pb_AB;
    float radius = fast_length((int3)(
        cx - consensus_set_x[contour_id][iA],
        cx - consensus_set_y[contour_id][iA], 0.0
    ));

    iter_center[get_local_id(0)] = (int2)(cx, cy);
    iter_radius[get_local_id(0)] = radius;

    if (cx < 0 || cy < 0) {
        printf("OCL:\n");
        printf("In Thread %d %d %d %f\n", get_local_id(0), cx, cy, radius);
        printf("\nA : %d %d\n", consensus_set_x[contour_id][iA], consensus_set_y[contour_id][iA] );
        printf("B : %d %d\n", consensus_set_x[contour_id][iB], consensus_set_y[contour_id][iB] );
        printf("C : %d %d\n", consensus_set_x[contour_id][iC], consensus_set_y[contour_id][iC] );
        printf("AB BC : %f %f\n", AB, BC );
    }

    votes[get_local_id(0)] = 0;
    for(int i = 0; i < consensus_size[contour_id]; i++) {
        if (fast_length((int3)(consensus_set_y[contour_id][i] - cy,
                               consensus_set_x[contour_id][i] - cx, 0.0)) -
            radius < radius_tolerance) {
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
                max_iter[i];
            }
        }

        
        center[contour_id] = iter_center[max_iter];
        radius[contour_id] = iter_radius[max_iter];
    }
}