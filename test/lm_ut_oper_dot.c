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
 * @file    lm_ut_oper_dot.c
 * @brief   Lin matrix unit test
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lm_ut_framework.h"
#include "lm_mat.h"
#include "lm_chk.h"
#include "lm_err.h"
#include "lm_shape.h"
#include "lm_oper_dot.h"


/*
 *******************************************************************************
 * Constant value definition
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Macros
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Data type definition
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Global variables
 *******************************************************************************
 */


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
 * function_example - Function example
 *
 * @param   [in]    input       Example input.
 * @param   [out]   *p_output   Example output.
 *
 * @return  [int]   Function executing result.
 * @retval  [0]     Success.
 * @retval  [-1]    Fail.
 *
 */
LM_UT_CASE_FUNC(lm_ut_oper_dot_gemm11)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_d1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -2.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        0.8,
    };
    lm_mat_elem_t mat_elem_c1[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d1[1 * 1] = {
        -2.24,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -0.326199,  -0.106995,  0.416017,   -0.370576,  -0.147158,
        -0.355147,  0.414011,   -0.218618,  -0.455724,  -0.144103,
        -0.090334,  0.038503,   -0.145152,  -0.471584,  -0.364228,
        -0.455494,  0.054250,   -0.443464,  -0.011128,  0.415689,
        -0.282315,  0.470647,   -0.367046,  0.462890,   -0.030941,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -0.422356,  0.157725,   0.163217,   -0.447589,  -0.096823,
        0.403186,   -0.202976,  0.106232,   0.153923,   0.092713,
        -0.447149,  -0.089780,  -0.022590,  0.107701,   -0.258303,
        -0.320906,  -0.471525,  -0.091808,  0.247264,   0.267083,
        -0.059708,  0.335479,   0.371599,   -0.440736,  -0.169760,
    };
    lm_mat_elem_t mat_elem_c2[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d2[5 * 5] = {
        0.036318204961000,  0.058285103303000,  -0.094667263287000, 0.147567263967000,  -0.159787708537000,
        0.569525393328000,  0.046119395492000,  -0.020755652432000, 0.149967569190000,  0.031987077266000,
        0.291663009638000,  0.091141431870000,  -0.099426386802000, 0.074648632075000,  -0.014311389615000,
        0.391299031656000,  0.101662019001000,  0.096927829697000,  -0.021497750444000, 0.090140593140000,
        0.326423214224000,  -0.335749051956000, -0.041783915890000, 0.287366111006000,  0.294661357524000,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_c3[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d3[3 * 3] = {
        21.150, 21.800, 22.450,
        36.650, 37.800, 38.950,
        52.150, 53.800, 55.450,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        -0.11,  0.12,  0.13,
        -0.21,  0.22,  0.23,
        -0.31,  0.32,  0.33,
        -0.41,  0.42,  0.43,
        -0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_c4[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d4[5 * 5] = {
        5.3400,     5.4800,     5.6200,     5.7600,     5.9000,
        9.4400,     9.6800,     9.9200,     10.1600,    10.4000,
        13.5400,    13.8800,    14.2200,    14.5600,    14.9000,
        17.6400,    18.0800,    18.5200,    18.9600,    19.4000,
        21.7400,    22.2800,    22.8200,    23.3600,    23.9000,
    };
    lm_mat_elem_t mat_elem_a5[1 * 3] = {
        0.11,  0.12,  0.13,
    };
    lm_mat_elem_t mat_elem_b5[3 * 1] = {
        -0.11,
        -0.21,
        -0.31,
    };
    lm_mat_elem_t mat_elem_c5[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d5[1 * 1] = {
        -0.077600,
    };
    lm_mat_elem_t mat_elem_a6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_b6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_c6[9 * 9] = {0};
    lm_mat_elem_t mat_elem_d6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b7[3 * 3] = {
        0.123, 0.456, 0.789,
        0.456, 0.123, 0.456,
        0.789, 0.456, 0.123
    };
    lm_mat_elem_t mat_elem_c7[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d7[3 * 3] = {
        32.130, 23.805, 30.798,
        45.810, 34.155, 44.478,
        59.490, 44.505, 58.158,
    };

    /*
     * Test [5 x 5] * [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test wrong output matrix
     * Test [5 x 5] * [5 x 5] => [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test [1 x 1] * [1 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c1,
                        sizeof(mat_elem_c1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d1,
                        sizeof(mat_elem_d1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test [5 x 5] * [5 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d2,
                        sizeof(mat_elem_d2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test [3 x 5] * [5 x 3] = [3 x 3]
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c3,
                        sizeof(mat_elem_c3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d3,
                        sizeof(mat_elem_d3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test [5 x 3] * [3 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c4,
                        sizeof(mat_elem_c4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d4,
                        sizeof(mat_elem_d4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test [1 x 3] * [3 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 3,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c5,
                        sizeof(mat_elem_c5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d5,
                        sizeof(mat_elem_d5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test [9 x 9] * [9 x 9] = [9 x 9]
     */
    result = lm_mat_set(&mat_a1, 9, 9,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 9, 9,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 9, 9,
                        mat_elem_c6,
                        sizeof(mat_elem_c6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 9, 9,
                        mat_elem_d6,
                        sizeof(mat_elem_d6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped [3 x 3] * [3 x 3] = [3 x 3] submatrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d7,
                        sizeof(mat_elem_d7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm11(&mat_a1_shaped, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

}

LM_UT_CASE_FUNC(lm_ut_oper_dot_gemm14)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_d1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -2.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        0.8,
    };
    lm_mat_elem_t mat_elem_c1[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d1[1 * 1] = {
        -2.24,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -0.326199,  -0.106995,  0.416017,   -0.370576,  -0.147158,
        -0.355147,  0.414011,   -0.218618,  -0.455724,  -0.144103,
        -0.090334,  0.038503,   -0.145152,  -0.471584,  -0.364228,
        -0.455494,  0.054250,   -0.443464,  -0.011128,  0.415689,
        -0.282315,  0.470647,   -0.367046,  0.462890,   -0.030941,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -0.422356,  0.157725,   0.163217,   -0.447589,  -0.096823,
        0.403186,   -0.202976,  0.106232,   0.153923,   0.092713,
        -0.447149,  -0.089780,  -0.022590,  0.107701,   -0.258303,
        -0.320906,  -0.471525,  -0.091808,  0.247264,   0.267083,
        -0.059708,  0.335479,   0.371599,   -0.440736,  -0.169760,
    };
    lm_mat_elem_t mat_elem_c2[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d2[5 * 5] = {
        0.036318204961000,  0.058285103303000,  -0.094667263287000, 0.147567263967000,  -0.159787708537000,
        0.569525393328000,  0.046119395492000,  -0.020755652432000, 0.149967569190000,  0.031987077266000,
        0.291663009638000,  0.091141431870000,  -0.099426386802000, 0.074648632075000,  -0.014311389615000,
        0.391299031656000,  0.101662019001000,  0.096927829697000,  -0.021497750444000, 0.090140593140000,
        0.326423214224000,  -0.335749051956000, -0.041783915890000, 0.287366111006000,  0.294661357524000,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_c3[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d3[3 * 3] = {
        21.150, 21.800, 22.450,
        36.650, 37.800, 38.950,
        52.150, 53.800, 55.450,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        -0.11,  0.12,  0.13,
        -0.21,  0.22,  0.23,
        -0.31,  0.32,  0.33,
        -0.41,  0.42,  0.43,
        -0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_c4[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d4[5 * 5] = {
        5.3400,     5.4800,     5.6200,     5.7600,     5.9000,
        9.4400,     9.6800,     9.9200,     10.1600,    10.4000,
        13.5400,    13.8800,    14.2200,    14.5600,    14.9000,
        17.6400,    18.0800,    18.5200,    18.9600,    19.4000,
        21.7400,    22.2800,    22.8200,    23.3600,    23.9000,
    };
    lm_mat_elem_t mat_elem_a5[1 * 3] = {
        0.11,  0.12,  0.13,
    };
    lm_mat_elem_t mat_elem_b5[3 * 1] = {
        -0.11,
        -0.21,
        -0.31,
    };
    lm_mat_elem_t mat_elem_c5[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d5[1 * 1] = {
        -0.077600,
    };
    lm_mat_elem_t mat_elem_a6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_b6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_c6[9 * 9] = {0};
    lm_mat_elem_t mat_elem_d6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b7[3 * 3] = {
        0.123, 0.456, 0.789,
        0.456, 0.123, 0.456,
        0.789, 0.456, 0.123
    };
    lm_mat_elem_t mat_elem_c7[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d7[3 * 3] = {
        32.130, 23.805, 30.798,
        45.810, 34.155, 44.478,
        59.490, 44.505, 58.158,
    };

    /*
     * Test [5 x 5] * [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test wrong output matrix
     * Test [5 x 5] * [5 x 5] => [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test [1 x 1] * [1 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c1,
                        sizeof(mat_elem_c1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d1,
                        sizeof(mat_elem_d1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test [5 x 5] * [5 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d2,
                        sizeof(mat_elem_d2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test [3 x 5] * [5 x 3] = [3 x 3]
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c3,
                        sizeof(mat_elem_c3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d3,
                        sizeof(mat_elem_d3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test [5 x 3] * [3 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c4,
                        sizeof(mat_elem_c4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d4,
                        sizeof(mat_elem_d4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test [1 x 3] * [3 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 3,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c5,
                        sizeof(mat_elem_c5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d5,
                        sizeof(mat_elem_d5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test [9 x 9] * [9 x 9] = [9 x 9]
     */
    result = lm_mat_set(&mat_a1, 9, 9,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 9, 9,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 9, 9,
                        mat_elem_c6,
                        sizeof(mat_elem_c6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 9, 9,
                        mat_elem_d6,
                        sizeof(mat_elem_d6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped [3 x 3] * [3 x 3] = [3 x 3] submatrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d7,
                        sizeof(mat_elem_d7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm14(&mat_a1_shaped, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

}

LM_UT_CASE_FUNC(lm_ut_oper_dot_gemm41)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_d1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -2.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        0.8,
    };
    lm_mat_elem_t mat_elem_c1[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d1[1 * 1] = {
        -2.24,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -0.326199,  -0.106995,  0.416017,   -0.370576,  -0.147158,
        -0.355147,  0.414011,   -0.218618,  -0.455724,  -0.144103,
        -0.090334,  0.038503,   -0.145152,  -0.471584,  -0.364228,
        -0.455494,  0.054250,   -0.443464,  -0.011128,  0.415689,
        -0.282315,  0.470647,   -0.367046,  0.462890,   -0.030941,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -0.422356,  0.157725,   0.163217,   -0.447589,  -0.096823,
        0.403186,   -0.202976,  0.106232,   0.153923,   0.092713,
        -0.447149,  -0.089780,  -0.022590,  0.107701,   -0.258303,
        -0.320906,  -0.471525,  -0.091808,  0.247264,   0.267083,
        -0.059708,  0.335479,   0.371599,   -0.440736,  -0.169760,
    };
    lm_mat_elem_t mat_elem_c2[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d2[5 * 5] = {
        0.036318204961000,  0.058285103303000,  -0.094667263287000, 0.147567263967000,  -0.159787708537000,
        0.569525393328000,  0.046119395492000,  -0.020755652432000, 0.149967569190000,  0.031987077266000,
        0.291663009638000,  0.091141431870000,  -0.099426386802000, 0.074648632075000,  -0.014311389615000,
        0.391299031656000,  0.101662019001000,  0.096927829697000,  -0.021497750444000, 0.090140593140000,
        0.326423214224000,  -0.335749051956000, -0.041783915890000, 0.287366111006000,  0.294661357524000,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_c3[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d3[3 * 3] = {
        21.150, 21.800, 22.450,
        36.650, 37.800, 38.950,
        52.150, 53.800, 55.450,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        -0.11,  0.12,  0.13,
        -0.21,  0.22,  0.23,
        -0.31,  0.32,  0.33,
        -0.41,  0.42,  0.43,
        -0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_c4[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d4[5 * 5] = {
        5.3400,     5.4800,     5.6200,     5.7600,     5.9000,
        9.4400,     9.6800,     9.9200,     10.1600,    10.4000,
        13.5400,    13.8800,    14.2200,    14.5600,    14.9000,
        17.6400,    18.0800,    18.5200,    18.9600,    19.4000,
        21.7400,    22.2800,    22.8200,    23.3600,    23.9000,
    };
    lm_mat_elem_t mat_elem_a5[1 * 3] = {
        0.11,  0.12,  0.13,
    };
    lm_mat_elem_t mat_elem_b5[3 * 1] = {
        -0.11,
        -0.21,
        -0.31,
    };
    lm_mat_elem_t mat_elem_c5[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d5[1 * 1] = {
        -0.077600,
    };
    lm_mat_elem_t mat_elem_a6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_b6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_c6[9 * 9] = {0};
    lm_mat_elem_t mat_elem_d6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b7[3 * 3] = {
        0.123, 0.456, 0.789,
        0.456, 0.123, 0.456,
        0.789, 0.456, 0.123
    };
    lm_mat_elem_t mat_elem_c7[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d7[3 * 3] = {
        32.130, 23.805, 30.798,
        45.810, 34.155, 44.478,
        59.490, 44.505, 58.158,
    };

    /*
     * Test [5 x 5] * [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test wrong output matrix
     * Test [5 x 5] * [5 x 5] => [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test [1 x 1] * [1 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c1,
                        sizeof(mat_elem_c1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d1,
                        sizeof(mat_elem_d1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test [5 x 5] * [5 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d2,
                        sizeof(mat_elem_d2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test [3 x 5] * [5 x 3] = [3 x 3]
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c3,
                        sizeof(mat_elem_c3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d3,
                        sizeof(mat_elem_d3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test [5 x 3] * [3 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c4,
                        sizeof(mat_elem_c4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d4,
                        sizeof(mat_elem_d4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test [1 x 3] * [3 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 3,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c5,
                        sizeof(mat_elem_c5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d5,
                        sizeof(mat_elem_d5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test [9 x 9] * [9 x 9] = [9 x 9]
     */
    result = lm_mat_set(&mat_a1, 9, 9,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 9, 9,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 9, 9,
                        mat_elem_c6,
                        sizeof(mat_elem_c6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 9, 9,
                        mat_elem_d6,
                        sizeof(mat_elem_d6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped [3 x 3] * [3 x 3] = [3 x 3] submatrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d7,
                        sizeof(mat_elem_d7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm41(&mat_a1_shaped, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

}

LM_UT_CASE_FUNC(lm_ut_oper_dot_gemm44)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_d1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -2.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        0.8,
    };
    lm_mat_elem_t mat_elem_c1[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d1[1 * 1] = {
        -2.24,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -0.326199,  -0.106995,  0.416017,   -0.370576,  -0.147158,
        -0.355147,  0.414011,   -0.218618,  -0.455724,  -0.144103,
        -0.090334,  0.038503,   -0.145152,  -0.471584,  -0.364228,
        -0.455494,  0.054250,   -0.443464,  -0.011128,  0.415689,
        -0.282315,  0.470647,   -0.367046,  0.462890,   -0.030941,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -0.422356,  0.157725,   0.163217,   -0.447589,  -0.096823,
        0.403186,   -0.202976,  0.106232,   0.153923,   0.092713,
        -0.447149,  -0.089780,  -0.022590,  0.107701,   -0.258303,
        -0.320906,  -0.471525,  -0.091808,  0.247264,   0.267083,
        -0.059708,  0.335479,   0.371599,   -0.440736,  -0.169760,
    };
    lm_mat_elem_t mat_elem_c2[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d2[5 * 5] = {
        0.036318204961000,  0.058285103303000,  -0.094667263287000, 0.147567263967000,  -0.159787708537000,
        0.569525393328000,  0.046119395492000,  -0.020755652432000, 0.149967569190000,  0.031987077266000,
        0.291663009638000,  0.091141431870000,  -0.099426386802000, 0.074648632075000,  -0.014311389615000,
        0.391299031656000,  0.101662019001000,  0.096927829697000,  -0.021497750444000, 0.090140593140000,
        0.326423214224000,  -0.335749051956000, -0.041783915890000, 0.287366111006000,  0.294661357524000,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_c3[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d3[3 * 3] = {
        21.150, 21.800, 22.450,
        36.650, 37.800, 38.950,
        52.150, 53.800, 55.450,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        -0.11,  0.12,  0.13,
        -0.21,  0.22,  0.23,
        -0.31,  0.32,  0.33,
        -0.41,  0.42,  0.43,
        -0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_c4[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d4[5 * 5] = {
        5.3400,     5.4800,     5.6200,     5.7600,     5.9000,
        9.4400,     9.6800,     9.9200,     10.1600,    10.4000,
        13.5400,    13.8800,    14.2200,    14.5600,    14.9000,
        17.6400,    18.0800,    18.5200,    18.9600,    19.4000,
        21.7400,    22.2800,    22.8200,    23.3600,    23.9000,
    };
    lm_mat_elem_t mat_elem_a5[1 * 3] = {
        0.11,  0.12,  0.13,
    };
    lm_mat_elem_t mat_elem_b5[3 * 1] = {
        -0.11,
        -0.21,
        -0.31,
    };
    lm_mat_elem_t mat_elem_c5[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d5[1 * 1] = {
        -0.077600,
    };
    lm_mat_elem_t mat_elem_a6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_b6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_c6[9 * 9] = {0};
    lm_mat_elem_t mat_elem_d6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b7[3 * 3] = {
        0.123, 0.456, 0.789,
        0.456, 0.123, 0.456,
        0.789, 0.456, 0.123
    };
    lm_mat_elem_t mat_elem_c7[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d7[3 * 3] = {
        32.130, 23.805, 30.798,
        45.810, 34.155, 44.478,
        59.490, 44.505, 58.158,
    };

    /*
     * Test [5 x 5] * [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test wrong output matrix
     * Test [5 x 5] * [5 x 5] => [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test [1 x 1] * [1 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c1,
                        sizeof(mat_elem_c1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d1,
                        sizeof(mat_elem_d1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test [5 x 5] * [5 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d2,
                        sizeof(mat_elem_d2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test [3 x 5] * [5 x 3] = [3 x 3]
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c3,
                        sizeof(mat_elem_c3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d3,
                        sizeof(mat_elem_d3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test [5 x 3] * [3 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c4,
                        sizeof(mat_elem_c4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d4,
                        sizeof(mat_elem_d4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test [1 x 3] * [3 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 3,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c5,
                        sizeof(mat_elem_c5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d5,
                        sizeof(mat_elem_d5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test [9 x 9] * [9 x 9] = [9 x 9]
     */
    result = lm_mat_set(&mat_a1, 9, 9,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 9, 9,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 9, 9,
                        mat_elem_c6,
                        sizeof(mat_elem_c6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 9, 9,
                        mat_elem_d6,
                        sizeof(mat_elem_d6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped [3 x 3] * [3 x 3] = [3 x 3] submatrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d7,
                        sizeof(mat_elem_d7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot_gemm44(&mat_a1_shaped, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

}

LM_UT_CASE_FUNC(lm_ut_oper_dot)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_d1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -2.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        0.8,
    };
    lm_mat_elem_t mat_elem_c1[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d1[1 * 1] = {
        -2.24,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -0.326199,  -0.106995,  0.416017,   -0.370576,  -0.147158,
        -0.355147,  0.414011,   -0.218618,  -0.455724,  -0.144103,
        -0.090334,  0.038503,   -0.145152,  -0.471584,  -0.364228,
        -0.455494,  0.054250,   -0.443464,  -0.011128,  0.415689,
        -0.282315,  0.470647,   -0.367046,  0.462890,   -0.030941,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -0.422356,  0.157725,   0.163217,   -0.447589,  -0.096823,
        0.403186,   -0.202976,  0.106232,   0.153923,   0.092713,
        -0.447149,  -0.089780,  -0.022590,  0.107701,   -0.258303,
        -0.320906,  -0.471525,  -0.091808,  0.247264,   0.267083,
        -0.059708,  0.335479,   0.371599,   -0.440736,  -0.169760,
    };
    lm_mat_elem_t mat_elem_c2[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d2[5 * 5] = {
        0.036318204961000,  0.058285103303000,  -0.094667263287000, 0.147567263967000,  -0.159787708537000,
        0.569525393328000,  0.046119395492000,  -0.020755652432000, 0.149967569190000,  0.031987077266000,
        0.291663009638000,  0.091141431870000,  -0.099426386802000, 0.074648632075000,  -0.014311389615000,
        0.391299031656000,  0.101662019001000,  0.096927829697000,  -0.021497750444000, 0.090140593140000,
        0.326423214224000,  -0.335749051956000, -0.041783915890000, 0.287366111006000,  0.294661357524000,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_c3[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d3[3 * 3] = {
        21.150, 21.800, 22.450,
        36.650, 37.800, 38.950,
        52.150, 53.800, 55.450,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        -0.11,  0.12,  0.13,
        -0.21,  0.22,  0.23,
        -0.31,  0.32,  0.33,
        -0.41,  0.42,  0.43,
        -0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_c4[5 * 5] = {0};
    lm_mat_elem_t mat_elem_d4[5 * 5] = {
        5.3400,     5.4800,     5.6200,     5.7600,     5.9000,
        9.4400,     9.6800,     9.9200,     10.1600,    10.4000,
        13.5400,    13.8800,    14.2200,    14.5600,    14.9000,
        17.6400,    18.0800,    18.5200,    18.9600,    19.4000,
        21.7400,    22.2800,    22.8200,    23.3600,    23.9000,
    };
    lm_mat_elem_t mat_elem_a5[1 * 3] = {
        0.11,  0.12,  0.13,
    };
    lm_mat_elem_t mat_elem_b5[3 * 1] = {
        -0.11,
        -0.21,
        -0.31,
    };
    lm_mat_elem_t mat_elem_c5[1 * 1] = {0};
    lm_mat_elem_t mat_elem_d5[1 * 1] = {
        -0.077600,
    };
    lm_mat_elem_t mat_elem_a6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_b6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_c6[9 * 9] = {0};
    lm_mat_elem_t mat_elem_d6[9 * 9] = {
        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
    };
    lm_mat_elem_t mat_elem_a7[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_c7[1 * 5] = {0};
    lm_mat_elem_t mat_elem_d7[1 * 5] = {
         2115, 2180, 2245, 2310, 2375,
    };
    lm_mat_elem_t mat_elem_a8[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b8[3 * 3] = {
        0.123, 0.456, 0.789,
        0.456, 0.123, 0.456,
        0.789, 0.456, 0.123
    };
    lm_mat_elem_t mat_elem_c8[3 * 3] = {0};
    lm_mat_elem_t mat_elem_d8[3 * 3] = {
        32.130, 23.805, 30.798,
        45.810, 34.155, 44.478,
        59.490, 44.505, 58.158,
    };

    /*
     * Test [5 x 5] * [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test wrong output matrix
     * Test [5 x 5] * [5 x 5] => [3 x 3] matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test [1 x 1] * [1 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c1,
                        sizeof(mat_elem_c1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d1,
                        sizeof(mat_elem_d1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test [5 x 5] * [5 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c2,
                        sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d2,
                        sizeof(mat_elem_d2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test [3 x 5] * [5 x 3] = [3 x 3]
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c3,
                        sizeof(mat_elem_c3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d3,
                        sizeof(mat_elem_d3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test [5 x 3] * [3 x 5] = [5 x 5]
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c4,
                        sizeof(mat_elem_c4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 5, 5,
                        mat_elem_d4,
                        sizeof(mat_elem_d4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test [1 x 3] * [3 x 1] = [1 x 1]
     */
    result = lm_mat_set(&mat_a1, 1, 3,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c5,
                        sizeof(mat_elem_c5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d5,
                        sizeof(mat_elem_d5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test [9 x 9] * [9 x 9] = [9 x 9]
     */
    result = lm_mat_set(&mat_a1, 9, 9,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 9, 9,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 9, 9,
                        mat_elem_c6,
                        sizeof(mat_elem_c6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 9, 9,
                        mat_elem_d6,
                        sizeof(mat_elem_d6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test [1 x 5] * [5 x 5] = [1 x 5]
     */
    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 5,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 5,
                        mat_elem_d7,
                        sizeof(mat_elem_d7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test reshaped [3 x 3] * [3 x 3] = [3 x 3] submatrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a8,
                        sizeof(mat_elem_a8) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b8,
                        sizeof(mat_elem_b8) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 3, 3,
                        mat_elem_c8,
                        sizeof(mat_elem_c8) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 3, 3,
                        mat_elem_d8,
                        sizeof(mat_elem_d8) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_dot(&mat_a1_shaped, &mat_b1, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

