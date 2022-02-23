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
 * @file    lm_ut_oper_axpy.c
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
#include "lm_oper_axpy.h"


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
LM_UT_CASE_FUNC(lm_ut_oper_axpy_if_alpha_is_zero)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_expect1 = {0};
    lm_mat_t mat_b1_shaped = {0};
    lm_mat_elem_t alpha1 = 0;

    /*
     * a1: Test 1 by 1 matrix
     * (alpha = 0)
     */
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_expect1[1 * 1] = {
        -1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 1,
                        mat_elem_expect1,
                        sizeof(mat_elem_expect1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_expect2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect2,
                        sizeof(mat_elem_expect2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_expect3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 3, 5,
                        mat_elem_expect3,
                        sizeof(mat_elem_expect3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_expect4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 3,
                        mat_elem_expect4,
                        sizeof(mat_elem_expect4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        -0.11,  0.12,  -0.13,   0.14,   -0.15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_expect5[1 * 5] = {
        11,  12,  13,  14,  15,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 5,
                        mat_elem_expect5,
                        sizeof(mat_elem_expect5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        -1.8,
        1.8,
        -1.8,
        1.8,
        -1.8,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_expect6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 1,
                        mat_elem_expect6,
                        sizeof(mat_elem_expect6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a7[3 * 3] = {
        22,     23,     -24,
        32,     -33,    34,
        -42,    43,     44,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
    };
    lm_mat_elem_t mat_elem_expect7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
    };

    result = lm_mat_set(&mat_a1,3, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b1, 1, 1, 3, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect7,
                        sizeof(mat_elem_expect7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

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
LM_UT_CASE_FUNC(lm_ut_oper_axpy_if_alpha_is_pos_1)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_expect1 = {0};
    lm_mat_t mat_b1_shaped = {0};
    lm_mat_elem_t alpha1 = LM_MAT_ONE_VAL;

    /*
     * a1: Test 1 by 1 matrix
     * (alpha = 0)
     */
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_expect1[1 * 1] = {
        -3.6,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 1,
                        mat_elem_expect1,
                        sizeof(mat_elem_expect1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_expect2[5 * 5] = {
        -22,    24,     26,     28,     30,
        42,     -44,    46,     48,     50,
        62,     64,     -66,    68,     70,
        82,     84,     86,     -88,    90,
        102,    104,    106,    108,    -110,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect2,
                        sizeof(mat_elem_expect2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_expect3[3 * 5] = {
        9.2000,     10.2000,    11.2000,    12.2000,    13.2000,
        19.2000,    20.2000,    21.2000,    22.2000,    23.2000,
        29.2000,    30.2000,    31.2000,    32.2000,    33.2000,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 3, 5,
                        mat_elem_expect3,
                        sizeof(mat_elem_expect3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_expect4[5 * 3] = {
        0.22,   0.24,   0.26,
        0.42,   0.44,   0.46,
        0.62,   0.64,   0.66,
        0.82,   0.84,   0.86,
        1.02,   1.04,   1.06,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 3,
                        mat_elem_expect4,
                        sizeof(mat_elem_expect4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        -0.11,  0.12,  -0.13,   0.14,   -0.15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_expect5[1 * 5] = {
        10.890, 12.12,  12.870, 14.14,  14.850,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 5,
                        mat_elem_expect5,
                        sizeof(mat_elem_expect5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        -1.8,
        1.8,
        -1.8,
        1.8,
        -1.8,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_expect6[5 * 1] = {
        9.2,
        22.80,
        29.20,
        42.80,
        49.20,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 1,
                        mat_elem_expect6,
                        sizeof(mat_elem_expect6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a7[3 * 3] = {
        22,     23,     -24,
        32,     -33,    34,
        -42,    43,     44,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
    };
    lm_mat_elem_t mat_elem_expect7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 44, 46, 0,  25,
        31, 64, 0,  68, 35,
        41, 0,  86, 88, 45,
        51, 52, 53, 54, 55,
    };

    result = lm_mat_set(&mat_a1,3, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b1, 1, 1, 3, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect7,
                        sizeof(mat_elem_expect7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

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
LM_UT_CASE_FUNC(lm_ut_oper_axpy_if_alpha_is_neg_1)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_expect1 = {0};
    lm_mat_t mat_b1_shaped = {0};
    lm_mat_elem_t alpha1 = (-LM_MAT_ONE_VAL);

    /*
     * a1: Test 1 by 1 matrix
     * (alpha = 0)
     */
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_expect1[1 * 1] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 1,
                        mat_elem_expect1,
                        sizeof(mat_elem_expect1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_expect2[5 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect2,
                        sizeof(mat_elem_expect2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_expect3[3 * 5] = {
        12.800, 13.800, 14.800, 15.800, 16.800,
        22.800, 23.800, 24.800, 25.800, 26.800,
        32.800, 33.800, 34.800, 35.800, 36.800,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 3, 5,
                        mat_elem_expect3,
                        sizeof(mat_elem_expect3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_expect4[5 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 3,
                        mat_elem_expect4,
                        sizeof(mat_elem_expect4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        -0.11,  0.12,  -0.13,   0.14,   -0.15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_expect5[1 * 5] = {
        11.11,  11.88,  13.13,  13.86,  15.15,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 5,
                        mat_elem_expect5,
                        sizeof(mat_elem_expect5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        -1.8,
        1.8,
        -1.8,
        1.8,
        -1.8,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_expect6[5 * 1] = {
        12.8,
        19.2,
        32.8,
        39.2,
        52.8,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 1,
                        mat_elem_expect6,
                        sizeof(mat_elem_expect6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a7[3 * 3] = {
        22,     23,     -24,
        32,     -33,    34,
        -42,    43,     44,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
    };
    lm_mat_elem_t mat_elem_expect7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 0,  0,  48, 25,
        31, 0,  66, 0,  35,
        41, 84, 0,  0,  45,
        51, 52, 53, 54, 55,
    };

    result = lm_mat_set(&mat_a1,3, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b1, 1, 1, 3, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect7,
                        sizeof(mat_elem_expect7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

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
LM_UT_CASE_FUNC(lm_ut_oper_axpy_if_alpha_is_neg_point_5)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_expect1 = {0};
    lm_mat_t mat_b1_shaped = {0};
    lm_mat_elem_t alpha1 = (-LM_MAT_ONE_VAL / 2);

    /*
     * a1: Test 1 by 1 matrix
     * (alpha = 0)
     */
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_expect1[1 * 1] = {
        -0.9,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 1,
                        mat_elem_expect1,
                        sizeof(mat_elem_expect1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_expect2[5 * 5] = {
        -5.5,   6.0,    6.5,    7.0,    7.5,
        10.5,   -11.0,  11.5,   12.0,   12.50,
        15.5,   16.0,   -16.5,  17.0,   17.5,
        20.5,   21.0,   21.5,   -22.0,  22.5,
        25.5,   26.0,   26.5,   27.0,   -27.5,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect2,
                        sizeof(mat_elem_expect2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_expect3[3 * 5] = {
        11.900, 12.900, 13.900, 14.900, 15.900,
        21.900, 22.900, 23.900, 24.900, 25.900,
        31.900, 32.900, 33.900, 34.900, 35.900,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 3, 5,
                        mat_elem_expect3,
                        sizeof(mat_elem_expect3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_expect4[5 * 3] = {
        0.055,  0.060,  0.065,
        0.105,  0.110,  0.115,
        0.155,  0.160,  0.165,
        0.205,  0.210,  0.215,
        0.255,  0.260,  0.265,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 3,
                        mat_elem_expect4,
                        sizeof(mat_elem_expect4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        -0.11,  0.12,  -0.13,   0.14,   -0.15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_expect5[1 * 5] = {
        11.055, 11.940, 13.065, 13.930, 15.075,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 5,
                        mat_elem_expect5,
                        sizeof(mat_elem_expect5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        -1.8,
        1.8,
        -1.8,
        1.8,
        -1.8,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_expect6[5 * 1] = {
        11.900,
        20.100,
        31.900,
        40.100,
        51.900,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 1,
                        mat_elem_expect6,
                        sizeof(mat_elem_expect6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a7[3 * 3] = {
        22,     23,     -24,
        32,     -33,    34,
        -42,    43,     44,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
    };
    lm_mat_elem_t mat_elem_expect7[5 * 5] = {
        11,     12,     13,     14,     15,
        21,     11.000, 11.500, 36.000, 25,
        31,     16.000, 49.500, 17.000, 35,
        41,     63.000, 21.500, 22.000, 45,
        51,     52,     53,     54,     55,
    };

    result = lm_mat_set(&mat_a1,3, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b1, 1, 1, 3, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect7,
                        sizeof(mat_elem_expect7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

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
LM_UT_CASE_FUNC(lm_ut_oper_axpy_if_alpha_is_pos_2)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_expect1 = {0};
    lm_mat_t mat_b1_shaped = {0};
    lm_mat_elem_t alpha1 = (LM_MAT_ONE_VAL * 2);

    /*
     * a1: Test 1 by 1 matrix
     * (alpha = 0)
     */
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_expect1[1 * 1] = {
        -5.4,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 1,
                        mat_elem_expect1,
                        sizeof(mat_elem_expect1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_expect2[5 * 5] = {
        -33,    36,     39,     42,     45,
        63,     -66,    69,     72,     75,
        93,     96,     -99,    102,    105,
        123,    126,    129,    -132,   135,
        153,    156,    159,    162,    -165,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect2,
                        sizeof(mat_elem_expect2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
        -1.8, -1.8, -1.8, -1.8, -1.8,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_expect3[3 * 5] = {
        7.4000,     8.4000,     9.4000,     10.4000,    11.4000,
        17.4000,    18.4000,    19.4000,    20.4000,    21.4000,
        27.4000,    28.4000,    29.4000,    30.4000,    31.4000,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 3, 5,
                        mat_elem_expect3,
                        sizeof(mat_elem_expect3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_expect4[5 * 3] = {
        0.33000,    0.36000,    0.39000,
        0.63000,    0.66000,    0.69000,
        0.93000,    0.96000,    0.99000,
        1.23000,    1.26000,    1.29000,
        1.53000,    1.56000,    1.59000,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 3,
                        mat_elem_expect4,
                        sizeof(mat_elem_expect4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        -0.11,  0.12,  -0.13,   0.14,   -0.15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_expect5[1 * 5] = {
        10.780, 12.240, 12.740, 14.280, 14.700,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 1, 5,
                        mat_elem_expect5,
                        sizeof(mat_elem_expect5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        -1.8,
        1.8,
        -1.8,
        1.8,
        -1.8,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_expect6[5 * 1] = {
        7.4000,
        24.6000,
        27.4000,
        44.6000,
        47.4000,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 1,
                        mat_elem_expect6,
                        sizeof(mat_elem_expect6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a7[3 * 3] = {
        22,     23,     -24,
        32,     -33,    34,
        -42,    43,     44,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        41, 42, 43, 44, 45,
        51, 52, 53, 54, 55,
    };
    lm_mat_elem_t mat_elem_expect7[5 * 5] = {
        11,     12,     13,     14,     15,
        21,     66,     69,     -24,    25,
        31,     96,     -33,    102,    35,
        41,     -42,    129,    132,    45,
        51,     52,     53,     54,     55,
    };

    result = lm_mat_set(&mat_a1,3, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b1, 1, 1, 3, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_expect1, 5, 5,
                        mat_elem_expect7,
                        sizeof(mat_elem_expect7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_axpy(alpha1, &mat_a1, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_expect1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

