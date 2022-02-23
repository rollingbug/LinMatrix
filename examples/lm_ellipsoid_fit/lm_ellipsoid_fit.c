/**
 *******************************************************************************
 * Copyright 2022 Y.H.Kuo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @noop    ___     ______  _   __
 * @noop   /  /    /_   _/ / \ / /  A lightweight matrix computation SW library.
 * @noop  /  /__  _ / /_  /   \ / __  Use at your own risk.
 * @noop /_____/ /_____/ /_/ \_/ /_/
 * @noop    ______    _____   _______  ____     ______ ___   ___
 * @noop   /      \  /  _  | /__  __/ / __ \   /_   _/ \  \ /  /
 * @noop  /  / /  / /  _   |  /  /   /  -- /  _ / /_   /  --  /
 * @noop /__/_/__/ /__/ \__| /__/   /__/ \_\ /_____/  /__/ \__\
 *
 * @file    lm_ellipsoid_fit.c
 * @brief   Lin matrix ellipsoid fitting example code
 * @note
 *
 *      This example code shows how to using Lin matrix library to
 *      find out a ellipsoid that fits the given samples based on
 *      least squares method.
 *
 *      The least squares equation and the constraint:
 *
 *          X' * X * Beta = lambda * Beta
 *          Beta * Beta   = 1
 *
 *      X contains:
 *      [
 *          x0^2  y0^2  z0^2  x0y0  x0z0  y0z0  x0  y0  z0  1;
 *          x1^2  y1^2  z1^2  x1y1  x1z1  y1z1  x1  y1  z1  1;
 *          ...
 *          xn^2  yn^2  zn^2  xnyn  xnzn  ynzn  xn  yn  zn  1;
 *      ]
 *
 *      Beta represents the coefficients [A, B, C, D, E, F, G, H, I, J]'
 *      of ellipsoid equation:
 *
 *          Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
 *
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "lm_lib.h"


/*
 *******************************************************************************
 * Constant value definition
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Data type definition
 *******************************************************************************
 */

#define LM_SAMPLES_NUM      85
#define LM_X_VECT_ELEM_NUM  10


/*
 *******************************************************************************
 * Global variables
 *******************************************************************************
 */

static const lm_mat_elem_t lm_elem_samples[LM_SAMPLES_NUM][3] =
{
    /*              x                       y                        z             */
    {    7.206477732793523e-01,   5.546558704453441e-01,  -6.234817813765182e-01    },
    {    6.821862348178138e-01,   8.198380566801620e-01,   8.299595141700405e-02    },
    {    4.008097165991903e-01,   8.380566801619433e-01,  -5.546558704453441e-01    },
    {    7.469635627530365e-01,   5.850202429149798e-01,  -6.012145748987854e-01    },
    {    1.761133603238866e-01,   3.137651821862348e-01,   1.032388663967611e-01    },
    {    4.149797570850203e-01,   9.170040485829961e-01,  -4.028340080971660e-01    },
    {    6.599190283400810e-01,   2.429149797570850e-01,  -5.910931174089069e-01    },
    {    8.886639676113360e-01,   4.959514170040486e-01,   2.024291497975709e-03    },
    {    2.773279352226721e-01,   9.008097165991904e-01,  -4.554655870445344e-01    },
    {    4.453441295546559e-01,   1.000000000000000e+00,  -1.821862348178138e-01    },
    {    7.226720647773279e-01,   3.906882591093118e-01,   1.680161943319838e-01    },
    {    7.651821862348178e-01,   4.048582995951417e-01,  -5.546558704453441e-01    },
    {    2.955465587044535e-01,   2.105263157894737e-01,  -5.546558704453441e-01    },
    {    2.591093117408907e-01,   7.692307692307693e-01,  -6.234817813765182e-01    },
    {    3.340080971659919e-01,   9.493927125506073e-01,  -5.870445344129555e-02    },
    {    2.550607287449393e-01,   1.821862348178138e-01,   4.048582995951417e-02    },
    {    4.230769230769231e-01,   5.445344129554657e-01,  -7.226720647773279e-01    },
    {    9.170040485829961e-01,   4.392712550607287e-01,  -1.923076923076923e-01    },
    {    4.493927125506073e-01,   8.502024291497975e-02,  -3.198380566801620e-01    },
    {    4.939271255060729e-01,   9.797570850202429e-01,  -4.251012145748988e-01    },
    {    3.421052631578947e-01,   5.870445344129555e-02,  -2.651821862348178e-01    },
    {    3.461538461538461e-01,   3.178137651821863e-01,  -6.376518218623483e-01    },
    {    6.943319838056680e-01,   2.894736842105263e-01,   1.234817813765182e-01    },
    {    3.724696356275304e-01,   1.295546558704453e-01,  -4.696356275303644e-01    },
    {    1.214574898785425e-02,   7.327935222672065e-01,  -1.983805668016194e-01    },
    {    4.089068825910931e-01,   1.700404858299595e-01,  -5.303643724696356e-01    },
    {    4.979757085020243e-01,   7.287449392712550e-02,  -1.417004048582996e-01    },
    {    1.619433198380567e-02,   6.882591093117409e-01,  -4.453441295546559e-02    },
    {    4.089068825910931e-01,   9.473684210526316e-01,  -6.477732793522267e-02    },
    {    5.000000000000000e-01,   6.781376518218624e-01,  -7.429149797570851e-01    },
    {    6.275303643724696e-02,   5.161943319838057e-01,   5.465587044534413e-02    },
    {    6.275303643724697e-01,   9.230769230769230e-01,  -1.012145748987854e-01    },
    {    6.255060728744940e-01,   6.943319838056680e-01,  -6.943319838056680e-01    },
    {    4.817813765182186e-01,   9.797570850202429e-01,  -2.793522267206478e-01    },
    {    9.311740890688260e-02,   8.097165991902835e-01,  -1.983805668016194e-01    },
    {    8.805668016194332e-01,   5.202429149797571e-01,   1.417004048582996e-02    },
    {    8.360323886639676e-01,   4.190283400809717e-01,  -4.271255060728745e-01    },
    {   -4.655870445344130e-02,   6.174089068825911e-01,  -3.987854251012146e-01    },
    {    2.388663967611336e-01,   6.558704453441296e-01,   1.781376518218624e-01    },
    {    9.392712550607287e-01,   5.060728744939271e-01,  -2.105263157894737e-01    },
    {    8.360323886639676e-01,   4.068825910931174e-01,  -4.878542510121457e-01    },
    {    4.109311740890689e-01,   4.676113360323887e-01,  -6.740890688259109e-01    },
    {    5.060728744939271e-01,   9.595141700404858e-01,  -6.882591093117409e-02    },
    {    3.522267206477733e-01,   4.777327935222672e-01,   2.510121457489878e-01    },
    {    7.307692307692308e-01,   6.457489878542511e-01,  -5.728744939271255e-01    },
    {    9.109311740890690e-02,   3.704453441295547e-01,  -5.202429149797571e-01    },
    {    2.186234817813765e-01,   8.704453441295548e-01,   1.214574898785425e-02    },
    {   -6.477732793522267e-02,   4.696356275303644e-01,  -2.995951417004049e-01    },
    {    7.834008097165992e-01,   2.044534412955465e-01,  -2.732793522267207e-01    },
    {    7.611336032388665e-01,   7.085020242914980e-01,  -5.546558704453441e-01    },
    {    8.421052631578948e-01,   6.963562753036437e-01,  -1.174089068825911e-01    },
    {    6.477732793522267e-02,   7.125506072874495e-01,  -2.874493927125507e-01    },
    {    5.263157894736843e-02,   7.004048582995952e-01,   2.429149797570850e-02    },
    {    5.182186234817814e-01,   1.336032388663968e-01,   4.048582995951417e-03    },
    {    2.408906882591093e-01,   3.947368421052632e-01,  -6.376518218623483e-01    },
    {    8.866396761133605e-01,   6.720647773279352e-01,  -3.076923076923077e-01    },
    {    8.218623481781377e-01,   8.056680161943320e-01,  -1.781376518218624e-01    },
    {    3.340080971659919e-01,   6.538461538461539e-01,   2.125506072874494e-01    },
    {    2.226720647773280e-01,   8.178137651821862e-01,   5.263157894736843e-02    },
    {    7.064777327935223e-01,   1.437246963562753e-01,  -3.340080971659919e-01    },
    {    5.364372469635628e-01,   2.085020242914980e-01,   1.153846153846154e-01    },
    {    4.230769230769231e-01,   9.210526315789475e-01,   1.821862348178137e-02    },
    {    5.688259109311741e-01,   7.469635627530365e-01,   1.842105263157895e-01    },
    {    6.801619433198380e-01,   9.068825910931175e-01,  -3.724696356275304e-01    },
    {    4.979757085020243e-01,   8.340080971659919e-01,   1.417004048582996e-01    },
    {    5.809716599190283e-01,   9.089068825910931e-01,  -4.433198380566802e-01    },
    {    4.028340080971660e-01,   9.655870445344130e-01,  -9.514170040485831e-02    },
    {    4.959514170040486e-01,   8.603238866396762e-01,  -5.809716599190283e-01    },
    {    2.246963562753037e-01,   2.408906882591093e-01,   8.704453441295547e-02    },
    {    4.048582995951417e-02,   2.591093117408907e-01,  -3.036437246963563e-01    },
    {    6.558704453441296e-01,   2.712550607287449e-01,   1.740890688259109e-01    },
    {    2.712550607287449e-01,   5.870445344129556e-01,   2.186234817813765e-01    },
    {   -4.048582995951417e-02,   5.951417004048584e-01,  -3.076923076923077e-01    },
    {    3.036437246963563e-01,   6.477732793522267e-02,  -1.376518218623482e-01    },
    {    3.097165991902834e-01,   2.975708502024292e-01,   1.720647773279352e-01    },
    {    4.048582995951417e-01,   2.267206477732794e-01,   1.619433198380567e-01    },
    {    2.732793522267207e-01,   4.068825910931174e-01,   2.246963562753037e-01    },
    {    7.489878542510121e-02,   2.692307692307692e-01,  -3.744939271255061e-01    },
    {    4.595141700404858e-01,   5.728744939271255e-01,  -7.105263157894737e-01    },
    {    3.906882591093118e-01,   1.153846153846154e-01,  -1.194331983805668e-01    },
    {    9.311740890688260e-02,   3.076923076923077e-01,  -4.777327935222672e-01    },
    {    2.732793522267207e-01,   7.064777327935223e-01,  -6.538461538461539e-01    },
    {    6.194331983805669e-01,   1.943319838056680e-01,   5.668016194331985e-02    },
    {    2.226720647773280e-01,   2.024291497975709e-01,   0.000000000000000e+00    },
    {    2.125506072874494e-01,   1.842105263157895e-01,  -6.072874493927126e-03    },
};


/*
 *******************************************************************************
 * Public functions declaration
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Private functions declaration
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Public functions
 *******************************************************************************
 */

/**
 * main - Lin matrix example code
 *
 * @return  [lm_rtn_t]  Function executing result.
 * @retval  [0]         Success.
 * @retval  [-N]        Failure (error code).
 *
 */
int32_t main()
{
    lm_rtn_t result;
    lm_mat_dim_size_t r_idx = 0;
    lm_mat_t vect_x = {0};
    lm_mat_t mat_a = {0};
    lm_mat_t mat_q = {0};
    lm_mat_t mat_beta = {0};
    lm_mat_t mat_work = {0};
    lm_mat_t mat_d_shaped = {0};
    lm_mat_t mat_sd_shaped = {0};

    lm_mat_elem_t elem_x_row_vect[LM_X_VECT_ELEM_NUM] = {0};
    lm_mat_elem_t elem_a[LM_X_VECT_ELEM_NUM * LM_X_VECT_ELEM_NUM] = {0};
    lm_mat_elem_t elem_q[LM_X_VECT_ELEM_NUM * LM_X_VECT_ELEM_NUM] = {0};
    lm_mat_elem_t elem_beta[LM_X_VECT_ELEM_NUM * 1] = {0};
    lm_mat_elem_t elem_work[LM_X_VECT_ELEM_NUM * 1] = {0};
    lm_mat_elem_t coeff[10];
    lm_mat_elem_t x;
    lm_mat_elem_t y;
    lm_mat_elem_t z;
    lm_mat_elem_t tmp_val;

    int32_t invert_sgn_det;
    lm_permute_list_t perm_lu = {0};
    lm_permute_elem_t perm_elem_lu[3 * 2] = {0};

    /* Setup required matrix handles for computation */
    result = lm_mat_set(&vect_x,
                        1,
                        LM_X_VECT_ELEM_NUM,
                        (lm_mat_elem_t *)elem_x_row_vect,
                        (sizeof(elem_x_row_vect) / sizeof(elem_x_row_vect[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_a,
                        LM_X_VECT_ELEM_NUM,
                        LM_X_VECT_ELEM_NUM,
                        (lm_mat_elem_t *)elem_a,
                        (sizeof(elem_a) / sizeof(elem_a[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_beta,
                        LM_X_VECT_ELEM_NUM,
                        1,
                        (lm_mat_elem_t *)elem_beta,
                        (sizeof(elem_beta) / sizeof(elem_beta[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work,
                        LM_X_VECT_ELEM_NUM,
                        1,
                        (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_q,
                        LM_X_VECT_ELEM_NUM,
                        LM_X_VECT_ELEM_NUM,
                        (lm_mat_elem_t *)elem_q,
                        (sizeof(elem_q) / sizeof(elem_q[0])));
    LM_RETURN_IF_ERR(result);

    /* Diagonal of matrix A */
    result = lm_shape_diag(&mat_a, 0, &mat_d_shaped);
    LM_RETURN_IF_ERR(result);

    /* Subdiagonal of matrix A */
    result = lm_shape_diag(&mat_a, -1, &mat_sd_shaped);
    LM_RETURN_IF_ERR(result);

    result = lm_oper_zeros(&mat_a);
    LM_RETURN_IF_ERR(result);

    /*
     * Compute X' * X, where X is equal to
     *
     *     -                                                      -
     *     | x0^2  y0^2  z0^2  x0*y0  x0*z0  y0*z0  x0  y0  z0  1 |
     *     | x1^2  y1^2  z1^2  x1*y1  x1*z1  y1*z1  x1  y1  z1  1 |
     * X = |   .                                                  |
     *     |   .                                                  |
     *     |   .                                                  |
     *     | xn^2  yn^2  zn^2  xn*yn  xn*zn  yn*zn  xn  yn  zn  1 |
     *     -                                                      -
     *
     * The matrix X requires 10 by n + 1 (e.g. 10 * 85) elements memory
     * space, it is not necessary to generate X first before computing
     * X' * X, an alternative way is to compute the [0,:]' * X[0,:], ... ,
     * X[n,:]' * X[n,:] sequentially and accumulate the result of each
     * multiplication.
     *
     * X[0] = [ x0^2, y0^2, z0^2, x0*y0, x0*z0, y0*z0, x0, y0, z0, 1]
     * X[1] = [ x1^2, y1^2, z1^2, x1*y1, x1*z1, y1*z1, x1, y1, z1, 1]
     *      .
     *      .
     *      .
     * X[n] = [ xn^2, yn^2, zn^2, xn*yn, xn*zn, yn*zn, xn, yn, zn, 1]
     *
     * A = X' * X = X[0]' * X[0] + X[1]' * X[1] + ... + X[n]' * X[n]
     *
     */
    for (r_idx = 0; r_idx < LM_SAMPLES_NUM; r_idx++) {

        x = lm_elem_samples[r_idx][0];
        y = lm_elem_samples[r_idx][1];
        z = lm_elem_samples[r_idx][2];

        /* Generate row vector of X */
        elem_x_row_vect[0] = x * x;
        elem_x_row_vect[1] = y * y;
        elem_x_row_vect[2] = z * z;
        elem_x_row_vect[3] = x * y;
        elem_x_row_vect[4] = x * z;
        elem_x_row_vect[5] = y * z;
        elem_x_row_vect[6] = x;
        elem_x_row_vect[7] = y;
        elem_x_row_vect[8] = z;
        elem_x_row_vect[9] = LM_MAT_ONE_VAL;

        /*
         * Accumulate X[0]' * X[0] + X[1]' * X[1] + ... + X[n]' * X[n]
         * and store the sum value in matrix A.
         */
        result = lm_oper_gemm(true, false, LM_MAT_ONE_VAL, &vect_x, &vect_x,
                              LM_MAT_ONE_VAL, &mat_a);
        LM_RETURN_IF_ERR(result);
    }

    /*
     * A = X' * X,
     * Now compute the Hessenberg similar matrix of A for later use
     */
    result = lm_symm_hess(&mat_a, &mat_beta, &mat_work);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work,
                        1,
                        LM_X_VECT_ELEM_NUM,
                        (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    /* Get the explicit Hessenberg similar matrix and transform matrix */
    result = lm_symm_hess_explicit(&mat_a, &mat_beta, &mat_q, &mat_work);
    LM_RETURN_IF_ERR(result);

    /* Compute the eigenvalues and eigenvectors of X' * X */
    result = lm_symm_eigen(&mat_d_shaped, &mat_sd_shaped, &mat_q);
    LM_RETURN_IF_ERR(result);

    /* Find out the minimum positive eigenvalue of X' * X */
    for (r_idx = 0; r_idx < LM_X_VECT_ELEM_NUM; r_idx++) {

        result = lm_mat_elem_get(&mat_d_shaped, r_idx, 0, &tmp_val);
        LM_RETURN_IF_ERR(result);

        if (tmp_val > LM_MAT_ZERO_VAL) {
            break;
        }
    }

    printf("\nMinimum eigenvalue: %e\n", tmp_val);

    /* Find out the corresponding eigenvector and store it in beta vector */
    result = lm_shape_col_vect(&mat_q, r_idx, &mat_q);
    LM_RETURN_IF_ERR(result);

    result = lm_oper_copy(&mat_q, &mat_beta);
    LM_RETURN_IF_ERR(result);

    printf("Corresponding eigenvector:\n");
    lm_mat_dump(&mat_beta);

    /* Compute the inverse of rotating and scaling matrix */
    result = lm_mat_clr(&mat_a);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_a,
                        3, 3,
                        (lm_mat_elem_t *)elem_a,
                        (sizeof(elem_a) / sizeof(elem_a[0])));
    LM_RETURN_IF_ERR(result);

    /*
     * Setup the rotating and scaling matrix and store it in matrix A
     *     -                     -
     *     |  b0    b3/2    b4/2 |
     * A = | b3/2    b1     b5/2 |
     *     | b4/2   b5/2     b2  |
     *     -                     -
     */
    elem_a[0] = (lm_mat_elem_t)(elem_beta[0]);
    elem_a[1] = (lm_mat_elem_t)(elem_beta[3] * 0.5);
    elem_a[2] = (lm_mat_elem_t)(elem_beta[4] * 0.5);
    elem_a[3] = (lm_mat_elem_t)(elem_beta[3] * 0.5);
    elem_a[4] = (lm_mat_elem_t)(elem_beta[1]);
    elem_a[5] = (lm_mat_elem_t)(elem_beta[5] * 0.5);
    elem_a[6] = (lm_mat_elem_t)(elem_beta[4] * 0.5);
    elem_a[7] = (lm_mat_elem_t)(elem_beta[5] * 0.5);
    elem_a[8] = (lm_mat_elem_t)(elem_beta[2]);

    /* Find the determinant of A */
    result = lm_permute_set(&perm_lu,
                            0,
                            perm_elem_lu,
                            (sizeof(perm_elem_lu) / sizeof(perm_elem_lu[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_lu_decomp(&mat_a, &perm_lu, &invert_sgn_det);
    LM_RETURN_IF_ERR(result);

    result = lm_lu_det(&mat_a, invert_sgn_det, &tmp_val);
    LM_RETURN_IF_ERR(result);

    /* Change the sign of eigenvectors if needed */
    if (tmp_val < 0) {
        result = lm_oper_scalar(&mat_beta, (-LM_MAT_ONE_VAL));
        LM_RETURN_IF_ERR(result);
    }

    /*
     * The content of matrix A has been destroy during LU decomposition,
     * so we should store the required elements in matrix A again.
     */
    elem_a[0] = (lm_mat_elem_t)(elem_beta[0]);
    elem_a[1] = (lm_mat_elem_t)(elem_beta[3] * 0.5);
    elem_a[2] = (lm_mat_elem_t)(elem_beta[4] * 0.5);
    elem_a[3] = (lm_mat_elem_t)(elem_beta[3] * 0.5);
    elem_a[4] = (lm_mat_elem_t)(elem_beta[1]);
    elem_a[5] = (lm_mat_elem_t)(elem_beta[5] * 0.5);
    elem_a[6] = (lm_mat_elem_t)(elem_beta[4] * 0.5);
    elem_a[7] = (lm_mat_elem_t)(elem_beta[5] * 0.5);
    elem_a[8] = (lm_mat_elem_t)(elem_beta[2]);

    /*
     * Store the coefficients for later use.
     */
    coeff[0] = elem_beta[0];
    coeff[1] = elem_beta[1];
    coeff[2] = elem_beta[2];
    coeff[3] = elem_beta[3];
    coeff[4] = elem_beta[4];
    coeff[5] = elem_beta[5];
    coeff[6] = elem_beta[6];
    coeff[7] = elem_beta[7];
    coeff[8] = elem_beta[8];
    coeff[9] = elem_beta[9];

    /*
     * The value should be approximately equal:
     *   (4.175213e-01)x^2 + (4.692406e-01)y^2 + (3.990643e-01)z^2
     * + (1.149918e-02)xy + (-2.540227e-02)xz + (1.498439e-02)yz
     * + (-3.825267e-01)x + (-4.952410e-01)y + (1.859542e-01)z
     * + (1.380762e-01) = 0
     */
    printf("The fitting ellipsoid for given samples:\n");
    printf("\t(%e)x^2 + (%e)y^2 + (%e)z^2 "
           "+ (%e)xy + (%e)xz + (%e)yz "
           "+ (%e)x + (%e)y + (%e)z + (%e) = 0\n",
           coeff[0], coeff[1], coeff[2], coeff[3], coeff[4],
           coeff[5], coeff[6], coeff[7], coeff[8], coeff[9]);

    /*
     * Start finding the square root of A.
     */

    /* Setup required matrix handles for computation again. */
    result = lm_mat_clr(&mat_d_shaped);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_sd_shaped);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_q);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_beta);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_work);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_beta,
                        mat_a.elem.dim.r,
                        1,
                        (lm_mat_elem_t *)elem_beta,
                        (sizeof(elem_beta) / sizeof(elem_beta[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work,
                        mat_a.elem.dim.r,
                        1,
                        (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_q,
                        mat_a.elem.dim.r,
                        mat_a.elem.dim.c,
                        (lm_mat_elem_t *)elem_q,
                        (sizeof(elem_q) / sizeof(elem_q[0])));
    LM_RETURN_IF_ERR(result);

    /* Diagonal of matrix A */
    result = lm_shape_diag(&mat_a, 0, &mat_d_shaped);
    LM_RETURN_IF_ERR(result);

    /* Subdiagonal of matrix A */
    result = lm_shape_diag(&mat_a, -1, &mat_sd_shaped);
    LM_RETURN_IF_ERR(result);

    /* Compute the Hessenberg similar matrix of A */
    result = lm_symm_hess(&mat_a, &mat_beta, &mat_work);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work,
                        1,
                        mat_a.elem.dim.c,
                        (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    /* Get the explicit Hessenberg similar matrix and transform matrix */
    result = lm_symm_hess_explicit(&mat_a, &mat_beta, &mat_q, &mat_work);
    LM_RETURN_IF_ERR(result);

    /* Compute the eigenvalues and eigenvectors of A */
    result = lm_symm_eigen(&mat_d_shaped, &mat_sd_shaped, &mat_q);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work,
                        mat_a.elem.dim.r,
                        mat_a.elem.dim.c,
                        (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    /* sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q, &mat_a, &mat_work);
    LM_RETURN_IF_ERR(result);

    /*
     * The value should be approximately equal to:
     *
     * -                                                -
     * |  6.4606726e-01   4.3624341e-03  -9.9602286e-03 |
     * |  4.3624341e-03   6.8497342e-01   5.7237223e-03 |
     * | -9.9602584e-03   5.7237186e-03   6.3161069e-01 |
     * -                                                -
     *
     */
    printf("\nInverse matrix to convert an ellipsoid to a sphere:\n");
    lm_mat_dump(&mat_a);

    /*
     * Compute the inverse of sqrtm(A)
     */
    result = lm_lu_decomp(&mat_a, &perm_lu, &invert_sgn_det);
    LM_RETURN_IF_ERR(result);

    result = lm_lu_invert(&mat_a, &perm_lu, &mat_q);
    LM_RETURN_IF_ERR(result);

    /* The center point offset of sphere */
    result = lm_mat_clr(&mat_work);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_beta);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work, 1, 3,
                        (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_beta, 1, 3,
                        (lm_mat_elem_t *)elem_beta,
                        (sizeof(elem_beta) / sizeof(elem_beta[0])));
    LM_RETURN_IF_ERR(result);

    elem_work[0] = (lm_mat_elem_t)(-coeff[6] * 0.5);
    elem_work[1] = (lm_mat_elem_t)(-coeff[7] * 0.5);
    elem_work[2] = (lm_mat_elem_t)(-coeff[8] * 0.5);

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_work, &mat_q,
                          LM_MAT_ZERO_VAL, &mat_beta);
    LM_RETURN_IF_ERR(result);

    /*
     * The value should be approximately equal to:
     *
     * -                                               -
     * | 2.9135677e-01   3.6086726e-01  -1.4588201e-01 |
     * -                                               -
     *
     */
    printf("\nThe center point offset of sphere:\n");
    lm_mat_dump(&mat_beta);

    /* Compute the radius of sphere */
    tmp_val = elem_beta[0] * elem_beta[0]
            + elem_beta[1] * elem_beta[1]
            + elem_beta[2] * elem_beta[2]
            - coeff[9];
    tmp_val = (lm_mat_elem_t)sqrt(tmp_val);

    /*
     * The value should be approximately equal to 3.135591e-01
     */
    printf("\nThe radius of sphere: %e\n", tmp_val);

    /* Validate the coefficients Beta * Beta = 1 */
    printf("\nThe coefficients Beta * Beta = %e\n",
             coeff[0] * coeff[0] + coeff[1] * coeff[1]
           + coeff[2] * coeff[2] + coeff[3] * coeff[3]
           + coeff[4] * coeff[4] + coeff[5] * coeff[5]
           + coeff[6] * coeff[6] + coeff[7] * coeff[7]
           + coeff[8] * coeff[8] + coeff[9] * coeff[9]);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

