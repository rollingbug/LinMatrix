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
 * @file    lm_ut_oper.c
 * @brief   Lin matrix unit test
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "lm_ut_mat.h"
#include "lm_ut_framework.h"
#include "lm_ut_oper_dot.h"
#include "lm_ut_oper_norm.h"
#include "lm_ut_oper_axpy.h"
#include "lm_ut_oper_gemm.h"
#include "lm_mat.h"
#include "lm_chk.h"
#include "lm_err.h"
#include "lm_shape.h"
#include "lm_oper.h"
#include "lm_oper_dot.h"
#include "lm_oper_norm.h"
#include "lm_oper_axpy.h"
#include "lm_oper_gemm.h"


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
LM_UT_CASE_FUNC(lm_ut_oper_zero)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_zero = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        100.0000000000,
    };
    lm_mat_elem_t mat_elem_a1_zero[1 * 1] = {
        0,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a2_zero[5 * 5] = {
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a3_zero[3 * 5] = {
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
    };
    lm_mat_elem_t mat_elem_a4_zero[5 * 3] = {
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,
    };
    lm_mat_elem_t mat_elem_a5[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a5_zero[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,   500.0000000000,
        100.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,   500.0000000000,
        100.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,   500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };

    /*
     * Test 1 by 1 matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_zero, 1, 1,
                        mat_elem_a1_zero, sizeof(mat_elem_a1_zero) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_zero, 5, 5,
                        mat_elem_a2_zero, sizeof(mat_elem_a2_zero) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 3 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_zero, 3, 5,
                        mat_elem_a3_zero, sizeof(mat_elem_a3_zero) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4, sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_zero, 5, 3,
                        mat_elem_a4_zero, sizeof(mat_elem_a4_zero) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix (set submatrix to zero)
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a5, sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_zero, 5, 5,
                        mat_elem_a5_zero, sizeof(mat_elem_a5_zero) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_zero);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_zeros_diagonal)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a_1_1: test 1 by 1 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_2: test 1 by 1 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_3: test 1 by 1 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_1: test 5 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_2: test 5 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_3: test 5 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_1: test 3 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_2: test 3 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_3: test 3 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_1: test 5 by 3 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_2: test 5 by 3 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0,
        0.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_3: test 5 by 3 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_1: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_2: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_3: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_diagonal(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
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
LM_UT_CASE_FUNC(lm_ut_zeros_triu)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a_1_1: test 1 by 1 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_2: test 1 by 1 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_3: test 1 by 1 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_1: test 5 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_2: test 5 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_3: test 5 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_1: test 3 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_2: test 3 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_3: test 3 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_1: test 5 by 3 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_2: test 5 by 3 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_3: test 5 by 3 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_1: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_2: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        -1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_3: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_triu(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
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
LM_UT_CASE_FUNC(lm_ut_zeros_tril)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a_1_1: test 1 by 1 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_2: test 1 by 1 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        (-1)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_3: test 1 by 1 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 1_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_1: test 5 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_2: test 5 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_2_3: test 5 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 2_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        (-1)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_1: test 3 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect,
                        ELEM_A_EXPECT_R, ELEM_A_EXPECT_C,
                        ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_2: test 3 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_3_3: test 3 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        (-1)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_1: test 5 by 3 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 1.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_2: test 5 by 3 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_4_3: test 5 by 3 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     3
    #define ZEROS_OFFSET        (-1)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0,
        0.0, 1.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_1: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_2: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_5_3: test 3 by 3 sub-matrix of 5 by 5 matrix (offset = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef ZEROS_OFFSET

    #define TEST_VAR(var)       var ## 5_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5
    #define ZEROS_OFFSET        (-1)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros_tril(&mat_a1_shaped, ZEROS_OFFSET);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
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
LM_UT_CASE_FUNC(lm_ut_oper_identity)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_identity = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        LM_MAT_ZERO_VAL,
    };
    lm_mat_elem_t mat_elem_a1_identity[1 * 1] = {
        LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
    };
    lm_mat_elem_t mat_elem_a2_identity[5 * 5] = {
        LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a3_identity[3 * 5] = {
        LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,
    };
    lm_mat_elem_t mat_elem_a4_identity[5 * 3] = {
        LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,
    };
    lm_mat_elem_t mat_elem_a5[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a5_identity[5 * 5] = {
        100.0000000000, 200.0000000000,     300.0000000000,     400.0000000000,     500.0000000000,
        100.0000000000, LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    500.0000000000,
        100.0000000000, LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     LM_MAT_ZERO_VAL,    500.0000000000,
        100.0000000000, LM_MAT_ZERO_VAL,    LM_MAT_ZERO_VAL,    LM_MAT_ONE_VAL,     500.0000000000,
        100.0000000000, 200.0000000000,     300.0000000000,     400.0000000000,     500.0000000000,
    };

    /*
     * Test 1 by 1 matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_identity, 1, 1,
                        mat_elem_a1_identity,
                        sizeof(mat_elem_a1_identity) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_identity, 5, 5,
                        mat_elem_a2_identity,
                        sizeof(mat_elem_a2_identity) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 3 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_identity, 3, 5,
                        mat_elem_a3_identity,
                        sizeof(mat_elem_a3_identity) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4, sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_identity, 5, 3,
                        mat_elem_a4_identity,
                        sizeof(mat_elem_a4_identity) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix (set submatrix to identity)
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a5, sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_identity, 5, 5,
                        mat_elem_a5_identity,
                        sizeof(mat_elem_a5_identity) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_identity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_abs)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -0.9,   -1.9,   -2.9,   -3.8,   -4.9,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        0.1,    0.2,    0.3,    0.4,    0.5,
        1.0,    2.0,    3.0,    4.0,    5.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        0.9,    1.9,    2.9,    3.8,    4.9,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        0.1,    0.2,    0.3,    0.4,    0.5,
        1.0,    2.0,    3.0,    4.0,    5.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.1,    -0.2,   0.3,
        -1.0,   2.0,    -3.0,
        0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,
        -0.9,   -1.9,   -2.9,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        0.1,    0.2,    0.3,
        1.0,    2.0,    3.0,
        0.0,    0.0,    0.0,
        1.1,    1.2,    1.3,
        0.9,    1.9,    2.9,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
    };
    lm_mat_elem_t mat_elem_b5[1 * 5] = {
        0.1,    0.2,    0.3,    0.4,    0.5,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        0.1,
        -1.0,
        0.0,
        1.1,
        -0.9,
    };
    lm_mat_elem_t mat_elem_b6[5 * 1] = {
        0.1,
        1.0,
        0.0,
        1.1,
        0.9,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 1,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -0.9,   -1.9,   -2.9,   -3.8,   -4.9,
    };
    lm_mat_elem_t mat_elem_b7[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    3.0,    4.0,    5.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -0.9,   -1.9,   -2.9,   -3.8,   -4.9,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b7,
                        sizeof(mat_elem_b7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_abs(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_oper_max)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test 1 by 1 matrix
     */
    lm_mat_elem_t value1 = 0.0;
    lm_mat_dim_size_t r_idx1 = 0;
    lm_mat_dim_size_t c_idx1 = 0;
    lm_mat_elem_t value_expected1 = -1.8;
    lm_mat_dim_size_t r_idx_expected1 = 0;
    lm_mat_dim_size_t c_idx_expected1 = 0;
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1, &r_idx1, &c_idx1, &value1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value1, value_expected1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx1 == r_idx_expected1), "");
    LM_UT_ASSERT((c_idx1 == c_idx_expected1), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t value2 = 0.0;
    lm_mat_dim_size_t r_idx2 = 0;
    lm_mat_dim_size_t c_idx2 = 0;
    lm_mat_elem_t value_expected2 = 5.0;
    lm_mat_dim_size_t r_idx_expected2 = 1;
    lm_mat_dim_size_t c_idx_expected2 = 4;
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1, &r_idx2, &c_idx2, &value2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value2, value_expected2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx2 == r_idx_expected2), "");
    LM_UT_ASSERT((c_idx2 == c_idx_expected2), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t value3 = 0.0;
    lm_mat_dim_size_t r_idx3 = 0;
    lm_mat_dim_size_t c_idx3 = 0;
    lm_mat_elem_t value_expected3 = 5.0;
    lm_mat_dim_size_t r_idx_expected3 = 1;
    lm_mat_dim_size_t c_idx_expected3 = 4;
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1, &r_idx3, &c_idx3, &value3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value3, value_expected3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx3 == r_idx_expected3), "");
    LM_UT_ASSERT((c_idx3 == c_idx_expected3), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t value4 = 0.0;
    lm_mat_dim_size_t r_idx4 = 0;
    lm_mat_dim_size_t c_idx4 = 0;
    lm_mat_elem_t value_expected4 = 2.0;
    lm_mat_dim_size_t r_idx_expected4 = 1;
    lm_mat_dim_size_t c_idx_expected4 = 1;
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.1,    -0.2,   0.3,
        -1.0,   2.0,    -3.0,
        0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,
        -1.1,   -2.1,   -3.1,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1, &r_idx4, &c_idx4, &value4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value4, value_expected4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx4 == r_idx_expected4), "");
    LM_UT_ASSERT((c_idx4 == c_idx_expected4), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t value5 = 0.0;
    lm_mat_dim_size_t r_idx5 = 0;
    lm_mat_dim_size_t c_idx5 = 0;
    lm_mat_elem_t value_expected5 = 0.5;
    lm_mat_dim_size_t r_idx_expected5 = 0;
    lm_mat_dim_size_t c_idx_expected5 = 4;
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1, &r_idx5, &c_idx5, &value5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value5, value_expected5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx5 == r_idx_expected5), "");
    LM_UT_ASSERT((c_idx5 == c_idx_expected5), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t value6 = 0.0;
    lm_mat_dim_size_t r_idx6 = 0;
    lm_mat_dim_size_t c_idx6 = 0;
    lm_mat_elem_t value_expected6 = 1.1;
    lm_mat_dim_size_t r_idx_expected6 = 3;
    lm_mat_dim_size_t c_idx_expected6 = 0;
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        0.1,
        -1.0,
        0.0,
        1.1,
        -1.1,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1, &r_idx6, &c_idx6, &value6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value6, value_expected6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx6 == r_idx_expected6), "");
    LM_UT_ASSERT((c_idx6 == c_idx_expected6), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a7: Test reshaped 3 by 3 matrix
     */

    lm_mat_elem_t value7 = 0.0;
    lm_mat_dim_size_t r_idx7 = 0;
    lm_mat_dim_size_t c_idx7 = 0;
    lm_mat_elem_t value_expected7 = 4.0;
    lm_mat_dim_size_t r_idx_expected7 = 0;
    lm_mat_dim_size_t c_idx_expected7 = 2;
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max(&mat_a1_shaped, &r_idx7, &c_idx7, &value7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value7, value_expected7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx7 == r_idx_expected7), "");
    LM_UT_ASSERT((c_idx7 == c_idx_expected7), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_max_abs)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test 1 by 1 matrix
     */
    lm_mat_elem_t value1 = 0.0;
    lm_mat_dim_size_t r_idx1 = 0;
    lm_mat_dim_size_t c_idx1 = 0;
    lm_mat_elem_t value_expected1 = 1.8;
    lm_mat_dim_size_t r_idx_expected1 = 0;
    lm_mat_dim_size_t c_idx_expected1 = 0;
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1, &r_idx1, &c_idx1, &value1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value1, value_expected1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx1 == r_idx_expected1), "");
    LM_UT_ASSERT((c_idx1 == c_idx_expected1), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t value2 = 0.0;
    lm_mat_dim_size_t r_idx2 = 0;
    lm_mat_dim_size_t c_idx2 = 0;
    lm_mat_elem_t value_expected2 = 5.1;
    lm_mat_dim_size_t r_idx_expected2 = 4;
    lm_mat_dim_size_t c_idx_expected2 = 4;
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1, &r_idx2, &c_idx2, &value2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value2, value_expected2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx2 == r_idx_expected2), "");
    LM_UT_ASSERT((c_idx2 == c_idx_expected2), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t value3 = 0.0;
    lm_mat_dim_size_t r_idx3 = 0;
    lm_mat_dim_size_t c_idx3 = 0;
    lm_mat_elem_t value_expected3 = 5.0;
    lm_mat_dim_size_t r_idx_expected3 = 1;
    lm_mat_dim_size_t c_idx_expected3 = 4;
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1, &r_idx3, &c_idx3, &value3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value3, value_expected3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx3 == r_idx_expected3), "");
    LM_UT_ASSERT((c_idx3 == c_idx_expected3), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t value4 = 0.0;
    lm_mat_dim_size_t r_idx4 = 0;
    lm_mat_dim_size_t c_idx4 = 0;
    lm_mat_elem_t value_expected4 = 3.1;
    lm_mat_dim_size_t r_idx_expected4 = 4;
    lm_mat_dim_size_t c_idx_expected4 = 2;
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.1,    -0.2,   0.3,
        -1.0,   2.0,    -3.0,
        0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,
        -1.1,   -2.1,   -3.1,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1, &r_idx4, &c_idx4, &value4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value4, value_expected4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx4 == r_idx_expected4), "");
    LM_UT_ASSERT((c_idx4 == c_idx_expected4), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t value5 = 0.0;
    lm_mat_dim_size_t r_idx5 = 0;
    lm_mat_dim_size_t c_idx5 = 0;
    lm_mat_elem_t value_expected5 = 0.5;
    lm_mat_dim_size_t r_idx_expected5 = 0;
    lm_mat_dim_size_t c_idx_expected5 = 4;
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1, &r_idx5, &c_idx5, &value5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value5, value_expected5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx5 == r_idx_expected5), "");
    LM_UT_ASSERT((c_idx5 == c_idx_expected5), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t value6 = 0.0;
    lm_mat_dim_size_t r_idx6 = 0;
    lm_mat_dim_size_t c_idx6 = 0;
    lm_mat_elem_t value_expected6 = 1.1;
    lm_mat_dim_size_t r_idx_expected6 = 3;
    lm_mat_dim_size_t c_idx_expected6 = 0;
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        0.1,
        -1.0,
        0.0,
        1.1,
        -1.1,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1, &r_idx6, &c_idx6, &value6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value6, value_expected6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx6 == r_idx_expected6), "");
    LM_UT_ASSERT((c_idx6 == c_idx_expected6), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a7: Test reshaped 3 by 3 matrix
     */

    lm_mat_elem_t value7 = 0.0;
    lm_mat_dim_size_t r_idx7 = 0;
    lm_mat_dim_size_t c_idx7 = 0;
    lm_mat_elem_t value_expected7 = 4.0;
    lm_mat_dim_size_t r_idx_expected7 = 0;
    lm_mat_dim_size_t c_idx_expected7 = 2;
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_max_abs(&mat_a1_shaped, &r_idx7, &c_idx7, &value7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value7, value_expected7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx7 == r_idx_expected7), "");
    LM_UT_ASSERT((c_idx7 == c_idx_expected7), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_min)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test 1 by 1 matrix
     */
    lm_mat_elem_t value1 = 0.0;
    lm_mat_dim_size_t r_idx1 = 0;
    lm_mat_dim_size_t c_idx1 = 0;
    lm_mat_elem_t value_expected1 = -1.8;
    lm_mat_dim_size_t r_idx_expected1 = 0;
    lm_mat_dim_size_t c_idx_expected1 = 0;
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1, &r_idx1, &c_idx1, &value1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value1, value_expected1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx1 == r_idx_expected1), "");
    LM_UT_ASSERT((c_idx1 == c_idx_expected1), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t value2 = 0.0;
    lm_mat_dim_size_t r_idx2 = 0;
    lm_mat_dim_size_t c_idx2 = 0;
    lm_mat_elem_t value_expected2 = -5.1;
    lm_mat_dim_size_t r_idx_expected2 = 4;
    lm_mat_dim_size_t c_idx_expected2 = 4;
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1, &r_idx2, &c_idx2, &value2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value2, value_expected2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx2 == r_idx_expected2), "");
    LM_UT_ASSERT((c_idx2 == c_idx_expected2), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t value3 = 0.0;
    lm_mat_dim_size_t r_idx3 = 0;
    lm_mat_dim_size_t c_idx3 = 0;
    lm_mat_elem_t value_expected3 = -3.0;
    lm_mat_dim_size_t r_idx_expected3 = 1;
    lm_mat_dim_size_t c_idx_expected3 = 2;
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1, &r_idx3, &c_idx3, &value3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value3, value_expected3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx3 == r_idx_expected3), "");
    LM_UT_ASSERT((c_idx3 == c_idx_expected3), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t value4 = 0.0;
    lm_mat_dim_size_t r_idx4 = 0;
    lm_mat_dim_size_t c_idx4 = 0;
    lm_mat_elem_t value_expected4 = -3.1;
    lm_mat_dim_size_t r_idx_expected4 = 4;
    lm_mat_dim_size_t c_idx_expected4 = 2;
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.1,    -0.2,   0.3,
        -1.0,   2.0,    -3.0,
        0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,
        -1.1,   -2.1,   -3.1,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1, &r_idx4, &c_idx4, &value4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value4, value_expected4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx4 == r_idx_expected4), "");
    LM_UT_ASSERT((c_idx4 == c_idx_expected4), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t value5 = 0.0;
    lm_mat_dim_size_t r_idx5 = 0;
    lm_mat_dim_size_t c_idx5 = 0;
    lm_mat_elem_t value_expected5 = -0.4;
    lm_mat_dim_size_t r_idx_expected5 = 0;
    lm_mat_dim_size_t c_idx_expected5 = 3;
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1, &r_idx5, &c_idx5, &value5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value5, value_expected5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx5 == r_idx_expected5), "");
    LM_UT_ASSERT((c_idx5 == c_idx_expected5), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t value6 = 0.0;
    lm_mat_dim_size_t r_idx6 = 0;
    lm_mat_dim_size_t c_idx6 = 0;
    lm_mat_elem_t value_expected6 = -1.1;
    lm_mat_dim_size_t r_idx_expected6 = 4;
    lm_mat_dim_size_t c_idx_expected6 = 0;
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        0.1,
        -1.0,
        0.0,
        1.1,
        -1.1,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1, &r_idx6, &c_idx6, &value6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value6, value_expected6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx6 == r_idx_expected6), "");
    LM_UT_ASSERT((c_idx6 == c_idx_expected6), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a7: Test reshaped 3 by 3 matrix
     */

    lm_mat_elem_t value7 = 0.0;
    lm_mat_dim_size_t r_idx7 = 0;
    lm_mat_dim_size_t c_idx7 = 0;
    lm_mat_elem_t value_expected7 = -3.0;
    lm_mat_dim_size_t r_idx_expected7 = 0;
    lm_mat_dim_size_t c_idx_expected7 = 1;
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min(&mat_a1_shaped, &r_idx7, &c_idx7, &value7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value7, value_expected7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx7 == r_idx_expected7), "");
    LM_UT_ASSERT((c_idx7 == c_idx_expected7), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_min_abs)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test 1 by 1 matrix
     */
    lm_mat_elem_t value1 = 0.0;
    lm_mat_dim_size_t r_idx1 = 0;
    lm_mat_dim_size_t c_idx1 = 0;
    lm_mat_elem_t value_expected1 = 1.8;
    lm_mat_dim_size_t r_idx_expected1 = 0;
    lm_mat_dim_size_t c_idx_expected1 = 0;
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1, &r_idx1, &c_idx1, &value1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value1, value_expected1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx1 == r_idx_expected1), "");
    LM_UT_ASSERT((c_idx1 == c_idx_expected1), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t value2 = 0.0;
    lm_mat_dim_size_t r_idx2 = 0;
    lm_mat_dim_size_t c_idx2 = 0;
    lm_mat_elem_t value_expected2 = 0.0;
    lm_mat_dim_size_t r_idx_expected2 = 2;
    lm_mat_dim_size_t c_idx_expected2 = 0;
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1, &r_idx2, &c_idx2, &value2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value2, value_expected2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx2 == r_idx_expected2), "");
    LM_UT_ASSERT((c_idx2 == c_idx_expected2), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t value3 = 0.0;
    lm_mat_dim_size_t r_idx3 = 0;
    lm_mat_dim_size_t c_idx3 = 0;
    lm_mat_elem_t value_expected3 = 0.0;
    lm_mat_dim_size_t r_idx_expected3 = 2;
    lm_mat_dim_size_t c_idx_expected3 = 0;
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1, &r_idx3, &c_idx3, &value3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value3, value_expected3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx3 == r_idx_expected3), "");
    LM_UT_ASSERT((c_idx3 == c_idx_expected3), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t value4 = 0.0;
    lm_mat_dim_size_t r_idx4 = 0;
    lm_mat_dim_size_t c_idx4 = 0;
    lm_mat_elem_t value_expected4 = 0.0;
    lm_mat_dim_size_t r_idx_expected4 = 2;
    lm_mat_dim_size_t c_idx_expected4 = 0;
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.1,    -0.2,   0.3,
        -1.0,   2.0,    -3.0,
        0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,
        -1.1,   -2.1,   -3.1,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1, &r_idx4, &c_idx4, &value4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value4, value_expected4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx4 == r_idx_expected4), "");
    LM_UT_ASSERT((c_idx4 == c_idx_expected4), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t value5 = 0.0;
    lm_mat_dim_size_t r_idx5 = 0;
    lm_mat_dim_size_t c_idx5 = 0;
    lm_mat_elem_t value_expected5 = 0.1;
    lm_mat_dim_size_t r_idx_expected5 = 0;
    lm_mat_dim_size_t c_idx_expected5 = 0;
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1, &r_idx5, &c_idx5, &value5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value5, value_expected5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx5 == r_idx_expected5), "");
    LM_UT_ASSERT((c_idx5 == c_idx_expected5), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t value6 = 0.0;
    lm_mat_dim_size_t r_idx6 = 0;
    lm_mat_dim_size_t c_idx6 = 0;
    lm_mat_elem_t value_expected6 = 0.0;
    lm_mat_dim_size_t r_idx_expected6 = 2;
    lm_mat_dim_size_t c_idx_expected6 = 0;
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
        0.1,
        -1.0,
        0.0,
        1.1,
        -1.1,
    };

    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1, &r_idx6, &c_idx6, &value6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value6, value_expected6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx6 == r_idx_expected6), "");
    LM_UT_ASSERT((c_idx6 == c_idx_expected6), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a7: Test reshaped 3 by 3 matrix
     */

    lm_mat_elem_t value7 = 0.0;
    lm_mat_dim_size_t r_idx7 = 0;
    lm_mat_dim_size_t c_idx7 = 0;
    lm_mat_elem_t value_expected7 = 0.0;
    lm_mat_dim_size_t r_idx_expected7 = 1;
    lm_mat_dim_size_t c_idx_expected7 = 0;
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        0.1,    -0.2,   0.3,    -0.4,   0.5,
        -1.0,   2.0,    -3.0,   4.0,    5.0,
        0.0,    -0.0,   0.0,    -0.0,   0.0,
        1.1,    1.2,    1.3,    1.4,    1.5,
        -1.1,   -2.1,   -3.1,   -4.1,   -5.1,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_min_abs(&mat_a1_shaped, &r_idx7, &c_idx7, &value7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(value7, value_expected7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((r_idx7 == r_idx_expected7), "");
    LM_UT_ASSERT((c_idx7 == c_idx_expected7), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_swap_row)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_swap = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a1_swap[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a2_no_change[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a2_swap[5 * 5] = {
        31,  32,  33,  34,  35, /* swap 0 to 2 */
        21,  22,  23,  24,  25,
        11,  12,  13,  14,  15,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a3_swap[3 * 5] = {
        31,  32,  33,  34,  35, /* swap 0 to 2 */
        21,  22,  23,  24,  25,
        11,  12,  13,  14,  15,

    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a4_swap[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        51,  52,  53, /* swap 3 to 4 */
        41,  42,  43,
    };
    lm_mat_elem_t mat_elem_a5[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a5_swap[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  42,  43,  44,  25, /* swap partial 1 to 3 */
        31,  32,  33,  34,  35,
        41,  22,  23,  24,  45,
        51,  52,  53,  54,  55,
    };

    /*
     * Test invalid source row index
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 100, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test invalid destination row index
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 2, 99);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test source index = destination index
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 5,
                        mat_elem_a2_no_change,
                        sizeof(mat_elem_a2_no_change) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 2, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 1 by 1 matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 1, 1,
                        mat_elem_a1_swap,
                        sizeof(mat_elem_a1_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 0, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 5,
                        mat_elem_a2_swap,
                        sizeof(mat_elem_a2_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 0, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 3 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 3, 5,
                        mat_elem_a3_swap,
                        sizeof(mat_elem_a3_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 0, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4, sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 3,
                        mat_elem_a4_swap,
                        sizeof(mat_elem_a4_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1, 3, 4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix (swap partially)
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a5, sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 5,
                        mat_elem_a5_swap,
                        sizeof(mat_elem_a5_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_row(&mat_a1_shaped, 0, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_swap_col)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_swap = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a1_swap[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a2_no_change[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a2_swap[5 * 5] = {
        /* swap 0 to 2 */
        13, 12, 11, 14, 15,
        23, 22, 21, 24, 25,
        33, 32, 31, 34, 35,
        43, 42, 41, 44, 45,
        53, 52, 51, 54, 55,

    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a3_swap[3 * 5] = {
        /* swap 0 to 2 */
        13, 12, 11, 14, 15,
        23, 22, 21, 24, 25,
        33, 32, 31, 34, 35,
    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a4_swap[5 * 3] = {
        /* swap 0 to 1 */
        12, 11, 13,
        22, 21, 23,
        32, 31, 33,
        42, 41, 43,
        52, 51, 53,
    };
    lm_mat_elem_t mat_elem_a5[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a5_swap[5 * 5] = {
        /* swap partial 1 to 3 */
        11,  12,  13,  14,  15,
        21,  24,  23,  22,  25,
        31,  34,  33,  32,  35,
        41,  44,  43,  42,  45,
        51,  52,  53,  54,  55,
    };

    /*
     * Test invalid source row index
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 100, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test invalid destination row index
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 2, 99);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test source index = destination index
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 5,
                        mat_elem_a2_no_change,
                        sizeof(mat_elem_a2_no_change) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 2, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 1 by 1 matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 1, 1,
                        mat_elem_a1_swap,
                        sizeof(mat_elem_a1_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 0, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 5,
                        mat_elem_a2_swap,
                        sizeof(mat_elem_a2_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 0, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 3 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 3, 5,
                        mat_elem_a3_swap,
                        sizeof(mat_elem_a3_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 0, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4, sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 3,
                        mat_elem_a4_swap,
                        sizeof(mat_elem_a4_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1, 0, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix (swap partially)
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a5, sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_swap, 5, 5,
                        mat_elem_a5_swap,
                        sizeof(mat_elem_a5_swap) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_swap_col(&mat_a1_shaped, 0, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_swap);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_permute_row)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_list_t perm_list_a1;

    /*
     * a1: Test invalid one line notation
     */
    lm_mat_elem_t mat_elem_a1_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };

    /* Create cycle notation manually */
    lm_permute_elem_t perm_elem_a1[2 * 3] = {0, 1, 7, 0xFFFF, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            4,
                            perm_elem_a1,
                            (sizeof(perm_elem_a1) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    perm_list_a1.elem.cyc_grp_num = 1;

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a1_expect,
                        sizeof(mat_elem_a1_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a2_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_elem_t perm_elem_a2[2 * 1] = {0, 0};

    result = lm_permute_set(&perm_list_a1,
                            1,
                            perm_elem_a2,
                            (sizeof(perm_elem_a2) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a2_expect,
                        sizeof(mat_elem_a2_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 3 matrix
     * One line notation: 0 1 2
     */
    lm_mat_elem_t mat_elem_a3_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a3[2 * 3] = {0, 1, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a3,
                            (sizeof(perm_elem_a3) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 3 by 3 matrix
     * One line notation: 0 2 1
     */
    lm_mat_elem_t mat_elem_a4_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  1,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a4[2 * 3] = {0, 2, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a4,
                            (sizeof(perm_elem_a4) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 3 matrix
     * One line notation: 1 0 2
     */
    lm_mat_elem_t mat_elem_a5_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a5[2 * 3] = {1, 0, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a5,
                            (sizeof(perm_elem_a5) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 3 by 3 matrix
     * One line notation: 1 2 0
     */
    lm_mat_elem_t mat_elem_a6_expect[3 * 3] = {
        0,  1,  0,
        0,  0,  1,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a6[2 * 3] = {1, 2, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a6,
                            (sizeof(perm_elem_a6) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test 3 by 3 matrix
     * One line notation: 2 0 1
     */
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  0,  1,
        1,  0,  0,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a7[2 * 3] = {2, 0, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a7,
                            (sizeof(perm_elem_a7) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test 3 by 3 matrix
     * One line notation: 2 1 0
     */
    lm_mat_elem_t mat_elem_a8_expect[3 * 3] = {
        0,  0,  1,
        0,  1,  0,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a8[2 * 3] = {2, 1, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a8,
                            (sizeof(perm_elem_a8) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a8_expect,
                        sizeof(mat_elem_a8_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a9: Test 2 by 2 submatrix 3 by 3 matrix
     * One line notation: 1 0
     */
    lm_mat_elem_t mat_elem_a9_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a9[2 * 2] = {1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            2,
                            perm_elem_a9,
                            (sizeof(perm_elem_a9) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a9_expect,
                        sizeof(mat_elem_a9_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row(&mat_a1_shaped, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_permute_row_inverse)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_list_t perm_list_a1;

    /*
     * a1: Test invalid one line notation
     */
    lm_mat_elem_t mat_elem_a1_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };

    /* Create cycle notation manually */
    lm_permute_elem_t perm_elem_a1[2 * 3] = {0, 1, 7, 0xFFFF, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            4,
                            perm_elem_a1,
                            (sizeof(perm_elem_a1) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    perm_list_a1.elem.cyc_grp_num = 1;

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a1_expect,
                        sizeof(mat_elem_a1_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a2_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_elem_t perm_elem_a2[2 * 1] = {0, 0};

    result = lm_permute_set(&perm_list_a1,
                            1,
                            perm_elem_a2,
                            (sizeof(perm_elem_a2) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a2_expect,
                        sizeof(mat_elem_a2_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 3 matrix
     * One line notation: 0 1 2
     *           inverse: 0 1 2
     */
    lm_mat_elem_t mat_elem_a3_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a3[2 * 3] = {0, 1, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a3,
                            (sizeof(perm_elem_a3) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 3 by 3 matrix
     * One line notation: 0 2 1
     *           inverse: 0 2 1
     */
    lm_mat_elem_t mat_elem_a4_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  1,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a4[2 * 3] = {0, 2, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a4,
                            (sizeof(perm_elem_a4) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 3 matrix
     * One line notation: 1 0 2
     *           inverse: 1 0 2
     */
    lm_mat_elem_t mat_elem_a5_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a5[2 * 3] = {1, 0, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a5,
                            (sizeof(perm_elem_a5) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 3 by 3 matrix
     * One line notation: 1 2 0
     *           inverse: 2 0 1
     */
    lm_mat_elem_t mat_elem_a6_expect[3 * 3] = {
        0,  0,  1,
        1,  0,  0,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a6[2 * 3] = {1, 2, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a6,
                            (sizeof(perm_elem_a6) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test 3 by 3 matrix
     * One line notation: 2 0 1
     *           inverse: 1 2 0
     */
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  1,  0,
        0,  0,  1,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a7[2 * 3] = {2, 0, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a7,
                            (sizeof(perm_elem_a7) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test 3 by 3 matrix
     * One line notation: 2 1 0
     *           inverse: 2 1 0
     */
    lm_mat_elem_t mat_elem_a8_expect[3 * 3] = {
        0,  0,  1,
        0,  1,  0,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a8[2 * 3] = {2, 1, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a8,
                            (sizeof(perm_elem_a8) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a8_expect,
                        sizeof(mat_elem_a8_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a9: Test 2 by 2 submatrix 3 by 3 matrix
     * One line notation: 1 0
                 inverse: 1 0
     */
    lm_mat_elem_t mat_elem_a9_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a9[2 * 2] = {1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            2,
                            perm_elem_a9,
                            (sizeof(perm_elem_a9) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a9_expect,
                        sizeof(mat_elem_a9_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_row_inverse(&mat_a1_shaped, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_row(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_permute_col)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_list_t perm_list_a1;

    /*
     * a1: Test invalid one line notation
     */
    lm_mat_elem_t mat_elem_a1_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };

    /* Create cycle notation manually */
    lm_permute_elem_t perm_elem_a1[2 * 3] = {0, 1, 7, 0xFFFF, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            4,
                            perm_elem_a1,
                            (sizeof(perm_elem_a1) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    perm_list_a1.elem.cyc_grp_num = 1;

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a1_expect,
                        sizeof(mat_elem_a1_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a2_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_elem_t perm_elem_a2[2 * 1] = {0, 0};

    result = lm_permute_set(&perm_list_a1,
                            1,
                            perm_elem_a2,
                            (sizeof(perm_elem_a2) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a2_expect,
                        sizeof(mat_elem_a2_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 3 matrix
     * One line notation: 0 1 2
     */
    lm_mat_elem_t mat_elem_a3_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a3[2 * 3] = {0, 1, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a3,
                            (sizeof(perm_elem_a3) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 3 by 3 matrix
     * One line notation: 0 2 1
     */
    lm_mat_elem_t mat_elem_a4_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  1,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a4[2 * 3] = {0, 2, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a4,
                            (sizeof(perm_elem_a4) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 3 matrix
     * One line notation: 1 0 2
     */
    lm_mat_elem_t mat_elem_a5_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a5[2 * 3] = {1, 0, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a5,
                            (sizeof(perm_elem_a5) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 3 by 3 matrix
     * One line notation: 1 2 0
     */
    lm_mat_elem_t mat_elem_a6_expect[3 * 3] = {
        0,  0,  1,
        1,  0,  0,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a6[2 * 3] = {1, 2, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a6,
                            (sizeof(perm_elem_a6) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test 3 by 3 matrix
     * One line notation: 2 0 1
     */
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  1,  0,
        0,  0,  1,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a7[2 * 3] = {2, 0, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a7,
                            (sizeof(perm_elem_a7) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test 3 by 3 matrix
     * One line notation: 2 1 0
     */
    lm_mat_elem_t mat_elem_a8_expect[3 * 3] = {
        0,  0,  1,
        0,  1,  0,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a8[2 * 3] = {2, 1, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a8,
                            (sizeof(perm_elem_a8) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a8_expect,
                        sizeof(mat_elem_a8_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a9: Test 2 by 2 submatrix 3 by 3 matrix
     * One line notation: 1 0
     */
    lm_mat_elem_t mat_elem_a9_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a9[2 * 2] = {1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            2,
                            perm_elem_a9,
                            (sizeof(perm_elem_a9) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a9_expect,
                        sizeof(mat_elem_a9_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col(&mat_a1_shaped, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_permute_col_inverse)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_list_t perm_list_a1;

    /*
     * a1: Test invalid one line notation
     */
    lm_mat_elem_t mat_elem_a1_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };

    /* Create cycle notation manually */
    lm_permute_elem_t perm_elem_a1[2 * 3] = {0, 1, 7, 0xFFFF, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            4,
                            perm_elem_a1,
                            (sizeof(perm_elem_a1) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    perm_list_a1.elem.cyc_grp_num = 1;

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a1_expect,
                        sizeof(mat_elem_a1_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a2_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_permute_elem_t perm_elem_a2[2 * 1] = {0, 0};

    result = lm_permute_set(&perm_list_a1,
                            1,
                            perm_elem_a2,
                            (sizeof(perm_elem_a2) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a2_expect,
                        sizeof(mat_elem_a2_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 3 matrix
     * One line notation: 0 1 2
     *           inverse: 0 1 2
     */
    lm_mat_elem_t mat_elem_a3_expect[3 * 3] = {
        1,  0,  0,
        0,  1,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a3[2 * 3] = {0, 1, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a3,
                            (sizeof(perm_elem_a3) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 3 by 3 matrix
     * One line notation: 0 2 1
     *           inverse: 0 2 1
     */
    lm_mat_elem_t mat_elem_a4_expect[3 * 3] = {
        1,  0,  0,
        0,  0,  1,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a4[2 * 3] = {0, 2, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a4,
                            (sizeof(perm_elem_a4) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 3 matrix
     * One line notation: 1 0 2
     *           inverse: 1 0 2
     */
    lm_mat_elem_t mat_elem_a5_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a5[2 * 3] = {1, 0, 2, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a5,
                            (sizeof(perm_elem_a5) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 3 by 3 matrix
     * One line notation: 1 2 0
     *           inverse: 2 0 1
     */
    lm_mat_elem_t mat_elem_a6_expect[3 * 3] = {
        0,  1,  0,
        0,  0,  1,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a6[2 * 3] = {1, 2, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a6,
                            (sizeof(perm_elem_a6) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test 3 by 3 matrix
     * One line notation: 2 0 1
     *           inverse: 1 2 0
     */
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  0,  1,
        1,  0,  0,
        0,  1,  0,
    };
    lm_permute_elem_t perm_elem_a7[2 * 3] = {2, 0, 1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a7,
                            (sizeof(perm_elem_a7) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test 3 by 3 matrix
     * One line notation: 2 1 0
     *           inverse: 2 1 0
     */
    lm_mat_elem_t mat_elem_a8_expect[3 * 3] = {
        0,  0,  1,
        0,  1,  0,
        1,  0,  0,
    };
    lm_permute_elem_t perm_elem_a8[2 * 3] = {2, 1, 0, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            3,
                            perm_elem_a8,
                            (sizeof(perm_elem_a8) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a8_expect,
                        sizeof(mat_elem_a8_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a9: Test 2 by 2 submatrix 3 by 3 matrix
     * One line notation: 1 0
     *           inverse: 1 0
     */
    lm_mat_elem_t mat_elem_a9_expect[3 * 3] = {
        0,  1,  0,
        1,  0,  0,
        0,  0,  1,
    };
    lm_permute_elem_t perm_elem_a9[2 * 2] = {1, 0, 0, 0};

    result = lm_permute_set(&perm_list_a1,
                            2,
                            perm_elem_a9,
                            (sizeof(perm_elem_a9) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_oline_to_cycle(&perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    lm_oper_identity(&mat_a1);

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a9_expect,
                        sizeof(mat_elem_a9_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_permute_col_inverse(&mat_a1_shaped, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Inverse again, should get an identity matrix */
    result = lm_oper_permute_col(&mat_a1, &perm_list_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(&perm_list_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_copy)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_copy = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Input wrong size output matrix
     */
    lm_mat_elem_t mat_elem_a1[3 * 1] = {
        11, 22, 33
    };
    lm_mat_elem_t mat_elem_a1_copy[1 * 3] = {
        0,
        0,
        0
    };

    result = lm_mat_set(&mat_a1, 3, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 3,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a2[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a2_copy[1 * 1] = {
        0,
    };
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 1,
                        mat_elem_a2_copy,
                        sizeof(mat_elem_a2_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a3[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a3_copy[5 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5,5,
                        mat_elem_a3_copy,
                        sizeof(mat_elem_a3_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 3 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a4[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a4_copy[3 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a4_copy,
                        sizeof(mat_elem_a4_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a5[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a5_copy[5 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a5_copy,
                        sizeof(mat_elem_a5_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Transpose the sub-matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a6_copy[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 3,
                        mat_elem_a6_copy,
                        sizeof(mat_elem_a6_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1_shaped, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_shaped, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_copy_diagonal)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_copy = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test if offset is out of range
     */
    lm_mat_elem_t mat_elem_a1[5 * 3] = {
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
    };
    lm_mat_elem_t mat_elem_a1_copy[5 * 3] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, -5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, 3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, 5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, -3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Input wrong size output matrix
     */
    lm_mat_elem_t mat_elem_a2[3 * 1] = {
        11, 22, 33
    };
    lm_mat_elem_t mat_elem_a2_copy[1 * 3] = {
        0,
        0,
        0
    };

    result = lm_mat_set(&mat_a1, 3, 1,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 3,
                        mat_elem_a2_copy,
                        sizeof(mat_elem_a2_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a3[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a3_copy[1 * 1] = {
        0,
    };
    lm_mat_elem_t mat_elem_a3_expect[1 * 1] = {
        11,
    };
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 1,
                        mat_elem_a3_copy,
                        sizeof(mat_elem_a3_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a4_copy[5 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a4_expect[5 * 5] = {
        11, 0,  0,  0,  0,
        0,  22, 0,  0,  0,
        0,  0,  33, 0,  0,
        0,  0,  0,  44, 0,
        0,  0,  0,  0,  55,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 5,
                        mat_elem_a4_copy,
                        sizeof(mat_elem_a4_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 5, 5,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a5_copy[3 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a5_expect[3 * 5] = {
        0,  12, 0,  0,  0,
        0,  0,  23, 0,  0,
        0,  0,  0,  34, 0,
    };
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a5_copy,
                        sizeof(mat_elem_a5_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 3, 5,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a6_copy[5 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a6_expect[5 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        41, 0,  0,
        0,  52, 0,
    };
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a6_copy,
                        sizeof(mat_elem_a6_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 5, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1, &mat_a1_copy, -3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Transpose the sub-matrix
     */
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a7_copy[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  0,  0,
        32, 0,  0,
        0,  43, 0,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 3,
                        mat_elem_a7_copy,
                        sizeof(mat_elem_a7_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_diagonal(&mat_a1_shaped, &mat_a1_copy, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_copy_triu)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_copy = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test if offset is out of range
     */
    lm_mat_elem_t mat_elem_a1[5 * 3] = {
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
    };
    lm_mat_elem_t mat_elem_a1_copy[5 * 3] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, 3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, -5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, 5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, -3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Input wrong size output matrix
     */
    lm_mat_elem_t mat_elem_a2[3 * 1] = {
        11, 22, 33
    };
    lm_mat_elem_t mat_elem_a2_copy[1 * 3] = {
        0,
        0,
        0
    };

    result = lm_mat_set(&mat_a1, 3, 1,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 3,
                        mat_elem_a2_copy,
                        sizeof(mat_elem_a2_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a3[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a3_copy[1 * 1] = {
        0,
    };
    lm_mat_elem_t mat_elem_a3_expect[1 * 1] = {
        11,
    };
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 1,
                        mat_elem_a3_copy,
                        sizeof(mat_elem_a3_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a4_copy[5 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a4_expect[5 * 5] = {
        11, 12, 13, 14, 15,
        0,  22, 23, 24, 25,
        0,  0,  33, 34, 35,
        0,  0,  0,  44, 45,
        0,  0,  0,  0,  55,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 5,
                        mat_elem_a4_copy,
                        sizeof(mat_elem_a4_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 5, 5,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a5_copy[3 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a5_expect[3 * 5] = {
        0,  12, 13, 14, 15,
        0,  0,  23, 24, 25,
        0,  0,  0,  34, 35,
    };
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a5_copy,
                        sizeof(mat_elem_a5_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 3, 5,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a6_copy[5 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a6_expect[5 * 3] = {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        0,  52, 53,
    };
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a6_copy,
                        sizeof(mat_elem_a6_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 5, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1, &mat_a1_copy, -3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Transpose the sub-matrix
     */
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a7_copy[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  0,  24,
        0,  0,  0,
        0,  0,  0,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 3,
                        mat_elem_a7_copy,
                        sizeof(mat_elem_a7_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_triu(&mat_a1_shaped, &mat_a1_copy, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_copy_tril)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_copy = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test if offset is out of range
     */
    lm_mat_elem_t mat_elem_a1[5 * 3] = {
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
        11, 22, 33,
    };
    lm_mat_elem_t mat_elem_a1_copy[5 * 3] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, 3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, -5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a1_copy,
                        sizeof(mat_elem_a1_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, 5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, -3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Input wrong size output matrix
     */
    lm_mat_elem_t mat_elem_a2[3 * 1] = {
        11, 22, 33
    };
    lm_mat_elem_t mat_elem_a2_copy[1 * 3] = {
        0,
        0,
        0
    };

    result = lm_mat_set(&mat_a1, 3, 1,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 3,
                        mat_elem_a2_copy,
                        sizeof(mat_elem_a2_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 1 by 1 matrix
     */
    lm_mat_elem_t mat_elem_a3[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a3_copy[1 * 1] = {
        0,
    };
    lm_mat_elem_t mat_elem_a3_expect[1 * 1] = {
        11,
    };
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 1, 1,
                        mat_elem_a3_copy,
                        sizeof(mat_elem_a3_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 1, 1,
                        mat_elem_a3_expect,
                        sizeof(mat_elem_a3_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 5 matrix
     */
    lm_mat_elem_t mat_elem_a4[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a4_copy[5 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a4_expect[5 * 5] = {
        11, 0,  0,  0,  0,
        21, 22, 0,  0,  0,
        31, 32, 33, 0,  0,
        41, 42, 43, 44, 0,
        51, 52, 53, 54, 55,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 5,
                        mat_elem_a4_copy,
                        sizeof(mat_elem_a4_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 5, 5,
                        mat_elem_a4_expect,
                        sizeof(mat_elem_a4_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 5 matrix
     */

    lm_mat_elem_t mat_elem_a5[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a5_copy[3 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a5_expect[3 * 5] = {
        0,  0,  0,  0,  0,
        21, 0,  0,  0,  0,
        31, 32, 0,  0,  0,
    };
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 5,
                        mat_elem_a5_copy,
                        sizeof(mat_elem_a5_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 3, 5,
                        mat_elem_a5_expect,
                        sizeof(mat_elem_a5_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 3 matrix
     */
    lm_mat_elem_t mat_elem_a6[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a6_copy[5 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a6_expect[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 5, 3,
                        mat_elem_a6_copy,
                        sizeof(mat_elem_a6_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 5, 3,
                        mat_elem_a6_expect,
                        sizeof(mat_elem_a6_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1, &mat_a1_copy, 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Transpose the sub-matrix
     */
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a7_copy[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
    };
    lm_mat_elem_t mat_elem_a7_expect[3 * 3] = {
        0,  0,  0,
        0,  0,  0,
        42, 0,  0,
    };
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_copy, 3, 3,
                        mat_elem_a7_copy,
                        sizeof(mat_elem_a7_copy) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, 3, 3,
                        mat_elem_a7_expect,
                        sizeof(mat_elem_a7_expect) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_tril(&mat_a1_shaped, &mat_a1_copy, -2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_expect, &mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
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
LM_UT_CASE_FUNC(lm_ut_oper_copy_transpose)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_out = {0};
    lm_mat_t mat_a1_transpose = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a1_out[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a1_transpose[1 * 1] = {
        11,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a2_out[5 * 5] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    lm_mat_elem_t mat_elem_a2_transpose[5 * 5] = {
        11,  21,  31,  41,  51,
        12,  22,  32,  42,  52,
        13,  23,  33,  43,  53,
        14,  24,  34,  44,  54,
        15,  25,  35,  45,  55,

    };
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_a3_out[5 * 3] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t mat_elem_a3_transpose[5 * 3] = {
        11,  21,  31,
        12,  22,  32,
        13,  23,  33,
        14,  24,  34,
        15,  25,  35,

    };
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        11,  12,  13,
        21,  22,  23,
        31,  32,  33,
        41,  42,  43,
        51,  52,  53,
    };
    lm_mat_elem_t mat_elem_a4_out[3 * 5] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

    };
    lm_mat_elem_t mat_elem_a4_transpose[3 * 5] = {
        11,  21,  31,  41,  51,
        12,  22,  32,  42,  52,
        13,  23,  33,  43,  53,
    };
    lm_mat_elem_t mat_elem_a5[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_a5_out[3 * 3] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t mat_elem_a5_transpose[3 * 3] = {
        22,  32,  42,
        23,  33,  43,
        24,  34,  44,
    };

    /*
     * Input wrong size output matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_out, 5, 5,
                        mat_elem_a2_out,
                        sizeof(mat_elem_a2_out) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_transpose(&mat_a1, &mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 1 by 1 matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_out, 1, 1,
                        mat_elem_a1_out,
                        sizeof(mat_elem_a1_out) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_transpose, 1, 1,
                        mat_elem_a1_transpose,
                        sizeof(mat_elem_a1_transpose) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_transpose(&mat_a1, &mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_out, &mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_out, 5, 5,
                        mat_elem_a2_out,
                        sizeof(mat_elem_a2_out) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_transpose, 5, 5,
                        mat_elem_a2_transpose,
                        sizeof(mat_elem_a2_transpose) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_transpose(&mat_a1, &mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_out, &mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 3 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_out, 5, 3,
                        mat_elem_a3_out,
                        sizeof(mat_elem_a3_out) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_transpose, 5, 3,
                        mat_elem_a3_transpose,
                        sizeof(mat_elem_a3_transpose) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_transpose(&mat_a1, &mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_out, &mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test 5 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_out, 3, 5,
                        mat_elem_a4_out,
                        sizeof(mat_elem_a4_out) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_transpose, 3, 5,
                        mat_elem_a4_transpose,
                        sizeof(mat_elem_a4_transpose) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_transpose(&mat_a1, &mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_out, &mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Transpose the sub-matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_out, 3, 3,
                        mat_elem_a5_out,
                        sizeof(mat_elem_a5_out) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_transpose, 3, 3,
                        mat_elem_a5_transpose,
                        sizeof(mat_elem_a5_transpose) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy_transpose(&mat_a1_shaped, &mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_out, &mat_a1_transpose);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_out);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_transpose);
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
LM_UT_CASE_FUNC(lm_ut_oper_transpose)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a_1_1: test 3 by 1 non-square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
        0.0,
        0.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_transpose(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_2: test 5 by 3 non-square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_transpose(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_1:
     *      test 1 by 1 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.0,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        -1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_transpose(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 3 by 3 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1,  2, -3,
         4,  5,  6,
        -7, -8,  9,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        -1, 4, -7,
         2, 5, -8,
        -3, 6,  9,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_transpose(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_3:
     *      test 5 by 5 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C

    #define TEST_VAR(var)       var ## 2_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        -1, -1, -1, -1, -1,
         2,  2,  2,  2,  2,
        -3, -3, -3, -3, -3,
         4,  4,  4,  4,  4,
        -5, -5, -5, -5, -5,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_transpose(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_4:
     *      test 3 by 3 sub-matrix of 5 by 5 orthogonal matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_EXPECT_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C

    #define TEST_VAR(var)       var ## 2_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_a_expect)
    #define ELEM_A_EXPECT_R     5
    #define ELEM_A_EXPECT_C     5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
        -1,  2, -3, 4, -5,
    };

    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        -1,  2, -3,  4, -5,
        -1,  2,  2,  2, -5,
        -1, -3, -3, -3, -5,
        -1,  4,  4,  4, -5,
        -1,  2, -3,  4, -5,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_transpose(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_expect);
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
LM_UT_CASE_FUNC(lm_ut_oper_scalar)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    lm_mat_elem_t scalar1 = 2;
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };
    lm_mat_elem_t mat_elem_b1[1 * 1] = {
        -3.6,
    };

    lm_mat_elem_t scalar2 = 0.1;
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -11,    12,     13,     14,     15,
        21,     -22,    23,     24,     25,
        31,     32,     -33,    34,     35,
        41,     42,     43,     -44,    45,
        51,     52,     53,     54,     -55,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        -1.1,   1.2,    1.3,    1.4,    1.5,
        2.1,    -2.2,   2.3,    2.4,    2.5,
        3.1,    3.2,    -3.3,   3.4,    3.5,
        4.1,    4.2,    4.3,    -4.4,   4.5,
        5.1,    5.2,    5.3,    5.4,    -5.5,
    };

    lm_mat_elem_t scalar3 = -10.0;
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };
    lm_mat_elem_t mat_elem_b3[3 * 5] = {
        -110,   -120,   -130,   -140,   -150,
        -210,   -220,   -230,   -240,   -250,
        -310,   -320,   -330,   -340,   -350,
    };

    lm_mat_elem_t scalar4 = -1.0;
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
        0.11,  0.12,  0.13,
        0.21,  0.22,  0.23,
        0.31,  0.32,  0.33,
        0.41,  0.42,  0.43,
        0.51,  0.52,  0.53,
    };
    lm_mat_elem_t mat_elem_b4[5 * 3] = {
        -0.11,  -0.12,  -0.13,
        -0.21,  -0.22,  -0.23,
        -0.31,  -0.32,  -0.33,
        -0.41,  -0.42,  -0.43,
        -0.51,  -0.52,  -0.53,
    };

    lm_mat_elem_t scalar5 = 0.0;
    lm_mat_elem_t mat_elem_a5[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b5[5 * 5] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };

    lm_mat_elem_t scalar6 = 0.0;
    lm_mat_elem_t mat_elem_a6[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };
    lm_mat_elem_t mat_elem_b6[5 * 5] = {
        11, 12, 13, 14, 15,
        21, 0,  0,  0,  25,
        31, 0,  0,  0,  35,
        41, 0,  0,  0,  45,
        51, 52, 53, 54, 55,
    };

    /*
     * a1: Test 1 by 1 matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 1, 1,
                        mat_elem_b1,
                        sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_scalar(&mat_a1, scalar1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2,
                        sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b2,
                        sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_scalar(&mat_a1, scalar2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 3, 5,
                        mat_elem_b3,
                        sizeof(mat_elem_b3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_scalar(&mat_a1, scalar3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a4,
                        sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 3,
                        mat_elem_b4,
                        sizeof(mat_elem_b4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_scalar(&mat_a1, scalar4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 5 by 5 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b5,
                        sizeof(mat_elem_b5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_scalar(&mat_a1, scalar5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test shaped 3 by 3 matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a6,
                        sizeof(mat_elem_a6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b6,
                        sizeof(mat_elem_b6) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_scalar(&mat_a1_shaped, scalar6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b1);
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
LM_UT_CASE_FUNC(lm_ut_oper_bandwidth)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_dim_size_t low_bw;
    lm_mat_dim_size_t up_bw;

    /*
     * a2_1:
     *      check 1 by 1 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      check 1 by 3 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            3
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      check 1 by 3 matrix (upper BW = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            3
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, -1.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      check 1 by 3 matrix (upper BW = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            3
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      2

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.1,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      check 3 by 1 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2:
     *      check 3 by 1 matrix (upper BW = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define EXPECTED_LOW_BW     1
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        1.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_3:
     *      check 1 by 3 matrix (upper BW = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define EXPECTED_LOW_BW     2
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        0.0,
       -1.1,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      check 2 by 3 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            3
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2:
     *      check 2 by 3 matrix (upper BW = 1, lower BW = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            3
    #define EXPECTED_LOW_BW     1
    #define EXPECTED_UP_BW      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_3:
     *      check 2 by 3 matrix (upper BW = 2, lower BW = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 5_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            3
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      2

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.1,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1:
     *      check 3 by 2 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 6_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            2
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_2:
     *      check 3 by 2 matrix (upper BW = 1, lower BW = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 6_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            2
    #define EXPECTED_LOW_BW     1
    #define EXPECTED_UP_BW      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, -1.0,
        1.0, 0.0,
        0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_3:
     *      check 2 by 3 matrix (upper BW = 1, lower BW = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 6_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            2
    #define EXPECTED_LOW_BW     2
    #define EXPECTED_UP_BW      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 1.0,
        0.1, 0.0,
        0.9, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_1:
     *      check 5 by 5 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 7_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define EXPECTED_LOW_BW     0
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_2:
     *      check 5 by 5 matrix (upper BW = 4, lower BW = 4)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 7_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define EXPECTED_LOW_BW     4
    #define EXPECTED_UP_BW      4

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
       -2.0, 0.0, 0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_3:
     *      check 5 by 5 matrix (upper BW = 1, lower BW = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef EXPECTED_LOW_BW
    #undef EXPECTED_UP_BW

    #define TEST_VAR(var)       var ## 7_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define EXPECTED_LOW_BW     1
    #define EXPECTED_UP_BW      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        2,     3,     0,     0,     0,
        1,    -2,    -3,     0,     0,
        0,    -1,     2,     3,     0,
        0,     0,     1,    -2,    -3,
        0,     0,     0,    -1,     2,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_oper_bandwidth(&mat_a1, &low_bw, &up_bw);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((low_bw == EXPECTED_LOW_BW), "");
    LM_UT_ASSERT((up_bw == EXPECTED_UP_BW), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_oper_givens)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_givens1 = {0};
    lm_mat_t mat_rotated1 = {0};
    lm_mat_elem_t adjacent_len;
    lm_mat_elem_t opposite_len;
    lm_mat_elem_t hypotenuse_len;
    lm_mat_elem_t sin_theta;
    lm_mat_elem_t cos_theta;

    /*
     * a2_1:
     *      test zero vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        0.0,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 0 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        5.0,
        0.0,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 90 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
       13.88,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      test -90 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
       -0.1,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_4:
     *      test 180 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 3_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.23456789,
         0.0,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      test 45 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.23456,
        1.23456,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2:
     *      test 135 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
       -1.23456,
        1.23456,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_3:
     *      test -45 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.23456,
       -1.23456,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_4:
     *      test -135 degree vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 4_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
       -1.23456,
       -1.23456,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      test very small degree vector (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
       8.88888,
       0.00111,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2:
     *      test very small degree vector (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0.0123,
       105.12345,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_3:
     *      test very small degree vector (case 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
     -18.88888,
       0.00111,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_4:
     *      test very small degree vector (case 4)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
     -18.88888,
      -0.00111,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_5:
     *      test very small degree vector (case 5)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_5
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
     201.123456,
      -0.000001,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_6:
     *      test very small degree vector (case 6)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_6
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
       0.000056,
      -0.000123,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_7:
     *      test very small degree vector (case 7)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_7
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
          -0.000000056,
      -10000.000123,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_8:
     *      test very small degree vector (case 7)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_GIVENS_R
    #undef ELEM_GIVENS_C
    #undef ELEM_GIVENS_NAME
    #undef ELEM_ROTATED_R
    #undef ELEM_ROTATED_C

    #define TEST_VAR(var)       var ## 5_8
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            1
    #define ELEM_GIVENS_NAME    TEST_VAR(mat_elem_givens)
    #define ELEM_GIVENS_R       2
    #define ELEM_GIVENS_C       2
    #define ELEM_ROTATED_NAME   TEST_VAR(mat_elem_rotated)
    #define ELEM_ROTATED_R      2
    #define ELEM_ROTATED_C      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -10000.000123,
            -0.000056,
    };
    lm_mat_elem_t ELEM_GIVENS_NAME[ELEM_GIVENS_R * ELEM_GIVENS_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_ROTATED_NAME[ELEM_ROTATED_R * ELEM_ROTATED_C] = {
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_givens1, ELEM_GIVENS_R, ELEM_GIVENS_C, ELEM_GIVENS_NAME,
                        (sizeof(ELEM_GIVENS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_rotated1, ELEM_ROTATED_R, ELEM_ROTATED_C, ELEM_ROTATED_NAME,
                        (sizeof(ELEM_ROTATED_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    adjacent_len = ELEM_A_NAME[0];
    opposite_len = ELEM_A_NAME[1];
    hypotenuse_len = sqrt(adjacent_len * adjacent_len + opposite_len * opposite_len);

    /* Compute the Givens rotation elements */
    result = lm_oper_givens(adjacent_len, opposite_len, &sin_theta, &cos_theta);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Complete the rotation matrix */
    ELEM_GIVENS_NAME[0] = cos_theta;
    ELEM_GIVENS_NAME[1] = sin_theta;
    ELEM_GIVENS_NAME[2] = -sin_theta;
    ELEM_GIVENS_NAME[3] = cos_theta;

    /* Perform the rotation */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_givens1, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_rotated1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* cos_theta ^ 2 + sin_theta ^ 2 = 1 */
    result = lm_chk_elem_almost_equal((sin_theta * sin_theta + cos_theta * cos_theta),
                               LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the first element of rotated vector should equal to length of hypotenuse */
    result = lm_chk_elem_almost_equal(fabs(ELEM_ROTATED_NAME[0]), hypotenuse_len);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The the second element of rotated vector should equal to zero */
    result = lm_chk_elem_almost_equal(ELEM_ROTATED_NAME[1], LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_givens1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_rotated1);
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
LM_UT_CASE_FUNC(lm_ut_oper_trace)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t trace1;

    /*
     * a2_1:
     *      test 1 by 1 zero matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_TRACE_EXPECTED (0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 1 by 1 matrix (value = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_TRACE_EXPECTED (1.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 1 by 5 matrix (value = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            5
    #define ELEM_TRACE_EXPECTED (-1.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.0,
        0.0,
        0.5,
        1.0,
        2.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 5 by 1 matrix (value = 10)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            1
    #define ELEM_TRACE_EXPECTED (10)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        10.0, 0.0, 0.5, 1.0, 2.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      test 3 by 2 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            2
    #define ELEM_TRACE_EXPECTED (5)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 2,
        3, 4,
        5, 6,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2:
     *      test 2 by 3 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            3
    #define ELEM_TRACE_EXPECTED (6)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 2, 3,
        4, 5, 6,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      test 5 by 5 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_TRACE_EXPECTED (2.386904000818371)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        2.484001553982729e-01,   1.095501912436015e-01,   4.528026444936933e-01,   1.014953102561324e-01,   4.779289456725224e-01,
        8.527923482724633e-01,   4.812078265072624e-01,   4.215047679898449e-01,   1.760255333193798e-01,   4.228461288480986e-01,
        8.265711139804156e-01,   5.195841581836095e-01,   4.934397805630906e-01,   2.987205683771679e-01,   8.667402415368496e-01,
        3.011900145922718e-01,   1.951226728777624e-01,   8.423802134237588e-01,   9.831674364232064e-01,   3.418239562986814e-01,
        1.774576041651304e-02,   7.924863910624095e-01,   4.814452368853818e-01,   8.789967178602316e-01,   1.806888019265385e-01,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2:
     *      test 5 by 5 matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_TRACE_EXPECTED (4.053026670734856e-01)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.579743574623941e-01,  -3.534361137890015e-01,   2.829025450565091e-01,  -3.127949710526705e-01,   1.662977345157027e-01,
        8.719619817154562e-01,   1.516009361074669e-01,  -1.070631991082344e-01,   1.584729794872919e-01,   1.131479025945137e-01,
        1.980717422622006e-01,   7.082071026375621e-01,  -4.688230721576792e-01,  -5.312453577770810e-01,  -1.079655874326820e-01,
        4.064726172421994e-01,  -4.972508867039845e-01,  -6.876183427865337e-01,  -5.875914540009397e-02,  -4.624677914817202e-01,
        3.018300499977347e-01,   3.207006944552704e-02,   5.828380873811846e-01,  -3.864712917207834e-01,   6.233095910613977e-01,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1:
     *      test 3 by 3 sub-matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_TRACE_EXPECTED

    #define TEST_VAR(var)       var ## 6_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_TRACE_EXPECTED (-1.592477785878181e-01)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.579743574623941e-01,  -3.534361137890015e-01,   2.829025450565091e-01,  -3.127949710526705e-01,   1.662977345157027e-01,
        8.719619817154562e-01,   1.516009361074669e-01,  -1.070631991082344e-01,   1.584729794872919e-01,   1.131479025945137e-01,
        1.980717422622006e-01,   7.082071026375621e-01,  -4.688230721576792e-01,  -5.312453577770810e-01,  -1.079655874326820e-01,
        4.064726172421994e-01,  -4.972508867039845e-01,  -6.876183427865337e-01,  -5.875914540009397e-02,  -4.624677914817202e-01,
        3.018300499977347e-01,   3.207006944552704e-02,   5.828380873811846e-01,  -3.864712917207834e-01,   6.233095910613977e-01,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_shape_submatrix(&mat_a1, 0, 0, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_a1_shaped, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, ELEM_TRACE_EXPECTED);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_oper_cases[] =
{
    {"lm_ut_oper_zero", lm_ut_oper_zero, NULL, NULL, 0, 0},
    {"lm_ut_zeros_diagonal", lm_ut_zeros_diagonal, NULL, NULL, 0, 0},
    {"lm_ut_zeros_triu", lm_ut_zeros_triu, NULL, NULL, 0, 0},
    {"lm_ut_zeros_tril", lm_ut_zeros_tril, NULL, NULL, 0, 0},

    {"lm_ut_oper_identity", lm_ut_oper_identity, NULL, NULL, 0, 0},
    {"lm_ut_oper_abs", lm_ut_oper_abs, NULL, NULL, 0, 0},

    {"lm_ut_oper_max", lm_ut_oper_max, NULL, NULL, 0, 0},
    {"lm_ut_oper_max_abs", lm_ut_oper_max_abs, NULL, NULL, 0, 0},
    {"lm_ut_oper_min", lm_ut_oper_min, NULL, NULL, 0, 0},
    {"lm_ut_oper_min_abs", lm_ut_oper_min_abs, NULL, NULL, 0, 0},

    {"lm_ut_oper_swap_row", lm_ut_oper_swap_row, NULL, NULL, 0, 0},
    {"lm_ut_oper_swap_col", lm_ut_oper_swap_col, NULL, NULL, 0, 0},

    {"lm_ut_oper_permute_row", lm_ut_oper_permute_row, NULL, NULL, 0, 0},
    {"lm_ut_oper_permute_row_inverse", lm_ut_oper_permute_row_inverse, NULL, NULL, 0, 0},

    {"lm_ut_oper_permute_col", lm_ut_oper_permute_col, NULL, NULL, 0, 0},
    {"lm_ut_oper_permute_col_inverse", lm_ut_oper_permute_col_inverse, NULL, NULL, 0, 0},

    {"lm_ut_oper_copy", lm_ut_oper_copy, NULL, NULL, 0, 0},
    {"lm_ut_oper_copy_diagonal", lm_ut_oper_copy_diagonal, NULL, NULL, 0, 0},
    {"lm_ut_oper_copy_triu", lm_ut_oper_copy_triu, NULL, NULL, 0, 0},
    {"lm_ut_oper_copy_tril", lm_ut_oper_copy_tril, NULL, NULL, 0, 0},
    {"lm_ut_oper_copy_transpose", lm_ut_oper_copy_transpose, NULL, NULL, 0, 0},

    {"lm_ut_oper_transpose", lm_ut_oper_transpose, NULL, NULL, 0, 0},

    {"lm_ut_oper_scalar", lm_ut_oper_scalar, NULL, NULL, 0, 0},

    {"lm_ut_oper_bandwidth", lm_ut_oper_bandwidth, NULL, NULL, 0, 0},

    {"lm_ut_oper_givens", lm_ut_oper_givens, NULL, NULL, 0, 0},

    {"lm_ut_oper_trace", lm_ut_oper_trace, NULL, NULL, 0, 0},

    {"lm_ut_oper_dot_gemm11", lm_ut_oper_dot_gemm11, NULL, NULL, 0, 0},
    {"lm_ut_oper_dot_gemm14", lm_ut_oper_dot_gemm14, NULL, NULL, 0, 0},
    {"lm_ut_oper_dot_gemm41", lm_ut_oper_dot_gemm41, NULL, NULL, 0, 0},
    {"lm_ut_oper_dot_gemm44", lm_ut_oper_dot_gemm44, NULL, NULL, 0, 0},
    {"lm_ut_oper_dot", lm_ut_oper_dot, NULL, NULL, 0, 0},

    {"lm_ut_oper_norm_fro", lm_ut_oper_norm_fro, NULL, NULL, 0, 0},

    {"lm_ut_oper_axpy_if_alpha_is_zero", lm_ut_oper_axpy_if_alpha_is_zero, NULL, NULL, 0, 0},
    {"lm_ut_oper_axpy_if_alpha_is_pos_1", lm_ut_oper_axpy_if_alpha_is_pos_1, NULL, NULL, 0, 0},
    {"lm_ut_oper_axpy_if_alpha_is_neg_1", lm_ut_oper_axpy_if_alpha_is_neg_1, NULL, NULL, 0, 0},
    {"lm_ut_oper_axpy_if_alpha_is_neg_point_5", lm_ut_oper_axpy_if_alpha_is_neg_point_5, NULL, NULL, 0, 0},
    {"lm_ut_oper_axpy_if_alpha_is_pos_2", lm_ut_oper_axpy_if_alpha_is_pos_2, NULL, NULL, 0, 0},

    {"lm_ut_oper_gemm_1by1", lm_ut_oper_gemm_1by1, NULL, NULL, 0, 0},
    {"lm_ut_oper_gemm_5by5", lm_ut_oper_gemm_5by5, NULL, NULL, 0, 0},
    {"lm_ut_oper_gemm_3by5", lm_ut_oper_gemm_3by5, NULL, NULL, 0, 0},
    {"lm_ut_oper_gemm_5by3", lm_ut_oper_gemm_5by3, NULL, NULL, 0, 0},
    {"lm_ut_oper_gemm_reshaped_5by1", lm_ut_oper_gemm_reshaped_5by1, NULL, NULL, 0, 0},

};

static lm_ut_suite_t lm_ut_oper_suites[] =
{
    {"lm_ut_oper_suites", lm_ut_oper_cases, sizeof(lm_ut_oper_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_list[] =
{
    {lm_ut_oper_suites, sizeof(lm_ut_oper_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_oper()
{
    lm_ut_run(lm_ut_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

