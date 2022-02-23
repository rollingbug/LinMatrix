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
 * @file    lm_ut_shape.c
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
LM_UT_CASE_FUNC(lm_ut_shape_row_vect)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};

    /*
     * Test invalid row index
     */
    lm_mat_elem_t mat_elem_a0[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a0,
                        sizeof(mat_elem_a0) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_a1, 6, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_row_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b2[1 * 5] = {
        31,     32,     -33,    34,     35,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_a1, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[1 * 5] = {
        31,  32,  33,  34,  35,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_a1, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[1 * 3] = {
        0.11,  0.12,  0.13,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
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

    result = lm_shape_row_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[1 * 1] = {
        21,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_a1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_shape_col_vect)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};

    /*
     * Test invalid row index
     */
    lm_mat_elem_t mat_elem_a0[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a0,
                        sizeof(mat_elem_a0) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_a1, 6, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_col_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b2[5 * 1] = {
        13,
        23,
        -33,
        43,
        53,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_a1, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[3 * 1] = {
        13,
        23,
        33,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_a1, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[5 * 1] = {
        0.11,
        0.21,
        0.31,
        0.41,
        0.51,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        11,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
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

    result = lm_shape_col_vect(&mat_a1, 0, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_shape_submatrix)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};

    /*
     * Test invalid row, column index and size
     */
    lm_mat_elem_t mat_elem_a0[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a0,
                        sizeof(mat_elem_a0) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 6, 0, 5, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 6, 5, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_shape_submatrix(&mat_a1, 4, 4, 5, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_shape_submatrix(&mat_a1, 4, 4, 1, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_submatrix(&mat_a1, 0, 0, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b2[2 * 5] = {
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 2, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 0, 2, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[3 * 3] = {
        12,  13,  14,
        22,  23,  24,
        32,  33,  34,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 3,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[2 * 2] = {
        0.42,  0.43,
        0.52,  0.53,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 2, 2,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 3, 1, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        15,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 4, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[2 * 1] = {
        21,
        31,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 2, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 0, 2, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test submatrix of submatrix
     */
    lm_mat_elem_t mat_elem_a7[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b7[2 * 2] = {
        0.21,  0.22,
        0.31,  0.32,
    };
    lm_mat_elem_t mat_elem_c7[1 * 1] = {
        0.32,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 2, 2,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 0, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1_shaped, 1, 1, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
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
LM_UT_CASE_FUNC(lm_ut_shape_diag_osf0)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_d1 = {0};
    lm_mat_t mat_e1 = {0};
    lm_mat_dim_offset_t diag_osf = 0;

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b2[5 * 1] = {
        -11,
        -22,
        -33,
        -44,
        -55,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[3 * 1] = {
        11,
        22,
        33,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[3 * 1] = {
        0.11,
        0.22,
        0.33,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        11,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[1 * 1] = {
        11,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test diagnoal of diagnoal of diagnoal
     */
    lm_mat_elem_t mat_elem_a7[3 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
    };
    lm_mat_elem_t mat_elem_b7[3 * 1] = {
        0.11,
        0.22,
        0.33,
    };
    lm_mat_elem_t mat_elem_c7[1 * 1] = {
        0.11,
    };
    lm_mat_elem_t mat_elem_d7[1 * 1] = {
        0.11,
    };
    lm_mat_elem_t mat_elem_e7[1 * 1] = {
        0.11,
    };

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, 1, 1,
                        mat_elem_c7,
                        sizeof(mat_elem_c7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_d1, 1, 1,
                        mat_elem_d7,
                        sizeof(mat_elem_d7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_e1, 1, 1,
                        mat_elem_e7,
                        sizeof(mat_elem_e7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1_shaped, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1_shaped, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1_shaped, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_e1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_e1);
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
LM_UT_CASE_FUNC(lm_ut_shape_diag_osf_pos1)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_dim_offset_t diag_osf = 1;

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
       -11,     12,     13,     14,     15,
        21,    -22,     23,     24,     25,
        31,     32,    -33,     34,     35,
        41,     42,     43,    -44,     45,
        51,     52,     53,     54,    -55,
    };
    lm_mat_elem_t mat_elem_b2[4 * 1] = {
        12,
        23,
        34,
        45,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 4, 1,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[3 * 1] = {
        12,
        23,
        34,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[2 * 1] = {
        0.12,
        0.23,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 2, 1,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        11,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[1 * 1] = {
        0,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_shape_diag_osf_pos2)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_dim_offset_t diag_osf = 2;

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
       -11,     12,     13,     14,     15,
        21,    -22,     23,     24,     25,
        31,     32,    -33,     34,     35,
        41,     42,     43,    -44,     45,
        51,     52,     53,     54,    -55,
    };
    lm_mat_elem_t mat_elem_b2[3 * 1] = {
        13,
        24,
        35,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[3 * 1] = {
        13,
        24,
        35,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[1 * 1] = {
        0.13,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        11,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[1 * 1] = {
        0,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_shape_diag_osf_neg1)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_dim_offset_t diag_osf = -1;

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
       -11,     12,     13,     14,     15,
        21,    -22,     23,     24,     25,
        31,     32,    -33,     34,     35,
        41,     42,     43,    -44,     45,
        51,     52,     53,     54,    -55,
    };
    lm_mat_elem_t mat_elem_b2[4 * 1] = {
        21,
        32,
        43,
        54,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 4, 1,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[2 * 1] = {
        21,
        32,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 2, 1,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[3 * 1] = {
        0.21,
        0.32,
        0.43,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        0,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[1 * 1] = {
        21,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_shape_diag_osf_neg2)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_dim_offset_t diag_osf = -2;

    /*
     * a1: Test 1 by 1 matrix
     */

    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
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

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
       -11,     12,     13,     14,     15,
        21,    -22,     23,     24,     25,
        31,     32,    -33,     34,     35,
        41,     42,     43,    -44,     45,
        51,     52,     53,     54,    -55,
    };
    lm_mat_elem_t mat_elem_b2[3 * 1] = {
        31,
        42,
        53,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[1 * 1] = {
        31,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
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
    lm_mat_elem_t mat_elem_b4[3 * 1] = {
        0.31,
        0.42,
        0.53,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 1,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };
    lm_mat_elem_t mat_elem_b5[1 * 1] = {
        0,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        11,
        21,
        31,
        41,
        51,
    };
    lm_mat_elem_t mat_elem_b6[1 * 1] = {
        31,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, diag_osf, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_shape_cases[] =
{
    {"lm_ut_shape_row_vect", lm_ut_shape_row_vect, NULL, NULL, 0, 0},
    {"lm_ut_shape_col_vect", lm_ut_shape_col_vect, NULL, NULL, 0, 0},
    {"lm_ut_shape_submatrix", lm_ut_shape_submatrix, NULL, NULL, 0, 0},
    {"lm_ut_shape_diag_osf0", lm_ut_shape_diag_osf0, NULL, NULL, 0, 0},
    {"lm_ut_shape_diag_osf_pos1", lm_ut_shape_diag_osf_pos1, NULL, NULL, 0, 0},
    {"lm_ut_shape_diag_osf_pos2", lm_ut_shape_diag_osf_pos2, NULL, NULL, 0, 0},
    {"lm_ut_shape_diag_osf_neg1", lm_ut_shape_diag_osf_neg1, NULL, NULL, 0, 0},
    {"lm_ut_shape_diag_osf_neg2", lm_ut_shape_diag_osf_neg2, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_shape_suites[] =
{
    {"lm_ut_shape_cases", lm_ut_shape_cases, sizeof(lm_ut_shape_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_list[] =
{
    {lm_ut_shape_suites, sizeof(lm_ut_shape_suites) / sizeof(lm_ut_suite_t), 0, 0}
};

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
int32_t lm_ut_run_shape()
{
    lm_ut_run(lm_ut_list);

    return 0;
}

/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

