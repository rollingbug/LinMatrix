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
 * @file    lm_ut_oper_gemm.c
 * @brief   Lin matrix unit test
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "lm_ut_framework.h"
#include "lm_mat.h"
#include "lm_chk.h"
#include "lm_err.h"
#include "lm_shape.h"
#include "lm_oper.h"
#include "lm_oper_gemm.h"
#include "lm_lu.h"


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
LM_UT_CASE_FUNC(lm_ut_oper_gemm_1by1)
{
    #undef GEMM_FUNC
    #define GEMM_FUNC(__is_transpose_a, __is_tramspose_b,\
                      __alpha, __mat_a, __mat_b, __beta, __mat_c) \
            lm_oper_gemm(__is_transpose_a, __is_tramspose_b, \
                         __alpha, __mat_a, __mat_b, __beta, __mat_c)

    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_c1_copy = {0};
    lm_mat_t mat_c1_expect = {0};

    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_B_NAME
    #undef ELEM_B_R
    #undef ELEM_B_C
    #undef ELEM_C_NAME
    #undef ELEM_C_R
    #undef ELEM_C_C
    #undef ELEM_C_COPY_NAME
    #undef ELEM_C_COPY_R
    #undef ELEM_C_COPY_C

    #define ELEM_A_NAME         mat_elem_a1
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_B_NAME         mat_elem_b1
    #define ELEM_B_R            1
    #define ELEM_B_C            1
    #define ELEM_C_NAME         mat_elem_c1
    #define ELEM_C_R            1
    #define ELEM_C_C            1
    #define ELEM_C_COPY_NAME    mat_elem_c1_copy
    #define ELEM_C_COPY_R       1
    #define ELEM_C_COPY_C       1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.1,
    };
    lm_mat_elem_t ELEM_B_NAME[ELEM_B_R * ELEM_B_C] = {
        0.2,
    };
    lm_mat_elem_t ELEM_C_NAME[ELEM_C_R * ELEM_C_C] = {
        0.3,
    };
    lm_mat_elem_t ELEM_C_COPY_NAME[ELEM_C_COPY_R * ELEM_C_COPY_C] = {
        0,
    };

    /*
     * a1_1:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_1
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_2
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.02,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_3:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_3
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.3,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_4:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_4
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.32,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_5:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_5
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.02,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_6:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_6
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.3,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_7:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_7
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.32,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_8:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_8
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.28,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_9:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_9
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.28,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_10:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_10
    #define ALPHA_MUL           (0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.6,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_11:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_11
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.04,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_12:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_12
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.64,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_13:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_13
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.56,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_14:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_14
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.56,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_15:
     *      A := A'
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_15
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.56,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_16:
     *      A := A
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_16
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.56,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_17:
     *      A := A'
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_17
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0.56,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
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
LM_UT_CASE_FUNC(lm_ut_oper_gemm_5by5)
{
    #undef GEMM_FUNC
    #define GEMM_FUNC(__is_transpose_a, __is_tramspose_b,\
                      __alpha, __mat_a, __mat_b, __beta, __mat_c) \
            lm_oper_gemm(__is_transpose_a, __is_tramspose_b, \
                         __alpha, __mat_a, __mat_b, __beta, __mat_c)

    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_c1_copy = {0};
    lm_mat_t mat_c1_expect = {0};

    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_B_NAME
    #undef ELEM_B_R
    #undef ELEM_B_C
    #undef ELEM_C_NAME
    #undef ELEM_C_R
    #undef ELEM_C_C
    #undef ELEM_C_COPY_NAME
    #undef ELEM_C_COPY_R
    #undef ELEM_C_COPY_C

    #define ELEM_A_NAME         mat_elem_a1
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_B_NAME         mat_elem_b1
    #define ELEM_B_R            5
    #define ELEM_B_C            5
    #define ELEM_C_NAME         mat_elem_c1
    #define ELEM_C_R            5
    #define ELEM_C_C            5
    #define ELEM_C_COPY_NAME    mat_elem_c1_copy
    #define ELEM_C_COPY_R       5
    #define ELEM_C_COPY_C       5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_B_NAME[ELEM_B_R * ELEM_B_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_C_NAME[ELEM_C_R * ELEM_C_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_C_COPY_NAME[ELEM_C_COPY_R * ELEM_C_COPY_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    /*
     * a1_1:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_1
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_2
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        21.15000000000000,   21.80000000000000,   22.45000000000000,   23.10000000000000,   23.75000000000000,
        36.65000000000000,   37.80000000000000,   38.95000000000000,   40.10000000000000,   41.25000000000000,
        52.15000000000000,   53.80000000000000,   55.45000000000000,   57.09999999999999,   58.75000000000000,
        67.65000000000001,   69.80000000000001,   71.95000000000000,   74.10000000000001,   76.25000000000000,
        83.15000000000001,   85.80000000000000,   88.45000000000000,   91.09999999999999,   93.75000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_3:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_3
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_4:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_4
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        22.25000000000000,   23.00000000000000,   23.75000000000000,   24.50000000000000,   25.25000000000000,
        38.75000000000000,   40.00000000000001,   41.24999999999999,   42.50000000000000,   43.75000000000000,
        55.25000000000000,   57.00000000000001,   58.74999999999999,   60.49999999999999,   62.25000000000000,
        71.75000000000000,   74.00000000000001,   76.25000000000000,   78.50000000000001,   80.75000000000000,
        88.25000000000000,   91.00000000000000,   93.75000000000000,   96.50000000000000,   99.25000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_5:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_5
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -21.15000000000000,  -21.80000000000000,  -22.45000000000000,  -23.10000000000000,  -23.75000000000000,
        -36.65000000000000,  -37.80000000000000,  -38.95000000000000,  -40.10000000000000,  -41.25000000000000,
        -52.15000000000000,  -53.80000000000000,  -55.45000000000000,  -57.09999999999999,  -58.75000000000000,
        -67.65000000000001,  -69.80000000000001,  -71.95000000000000,  -74.10000000000001,  -76.25000000000000,
        -83.15000000000001,  -85.80000000000000,  -88.45000000000000,  -91.09999999999999,  -93.75000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_6:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_6
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -1.1, -1.2, -1.3, -1.4, -1.5,
        -2.1, -2.2, -2.3, -2.4, -2.5,
        -3.1, -3.2, -3.3, -3.4, -3.5,
        -4.1, -4.2, -4.3, -4.4, -4.5,
        -5.1, -5.2, -5.3, -5.4, -5.5,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_7:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_7
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -22.25000000000000,  -23.00000000000000,  -23.75000000000000,  -24.50000000000000,  -25.25000000000000,
        -38.75000000000000,  -40.00000000000001,  -41.24999999999999,  -42.50000000000000,  -43.75000000000000,
        -55.25000000000000,  -57.00000000000001,  -58.74999999999999,  -60.49999999999999,  -62.25000000000000,
        -71.75000000000000,  -74.00000000000001,  -76.25000000000000,  -78.50000000000001,  -80.75000000000000,
        -88.25000000000000,  -91.00000000000000,  -93.75000000000000,  -96.50000000000000,  -99.25000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_8:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_8
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        20.05000000000000,   20.60000000000000,   21.15000000000000,   21.70000000000000,   22.25000000000000,
        34.55000000000000,   35.60000000000000,   36.65000000000000,   37.70000000000000,   38.75000000000000,
        49.05000000000000,   50.60000000000000,   52.15000000000000,   53.70000000000000,   55.25000000000000,
        63.55000000000000,   65.60000000000001,   67.65000000000001,   69.70000000000000,   71.75000000000000,
        78.05000000000001,   80.59999999999999,   83.15000000000001,   85.69999999999999,   88.25000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_9:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_9
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -20.05000000000000,  -20.60000000000000,  -21.15000000000000,  -21.70000000000000,  -22.25000000000000,
        -34.55000000000000,  -35.60000000000000,  -36.65000000000000,  -37.70000000000000,  -38.75000000000000,
        -49.05000000000000,  -50.60000000000000,  -52.15000000000000,  -53.70000000000000,  -55.25000000000000,
        -63.55000000000000,  -65.60000000000001,  -67.65000000000001,  -69.70000000000000,  -71.75000000000000,
        -78.05000000000001,  -80.59999999999999,  -83.15000000000001,  -85.69999999999999,  -88.25000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_10:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_10
    #define ALPHA_MUL           (0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -2.200000000000000,  -2.400000000000000,  -2.600000000000000,  -2.800000000000000,  -3.000000000000000,
        -4.200000000000000,  -4.400000000000000,  -4.600000000000000,  -4.800000000000000,  -5.000000000000000,
        -6.200000000000000,  -6.400000000000000,  -6.600000000000000,  -6.800000000000000,  -7.000000000000000,
        -8.199999999999999,  -8.400000000000000,  -8.600000000000000,  -8.800000000000001,  -9.000000000000000,
        -10.20000000000000,  -10.40000000000000,  -10.60000000000000,  -10.80000000000000,  -11.00000000000000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_11:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_11
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
         42.300,    43.600,    44.900,    46.200,    47.500,
         73.300,    75.600,    77.900,    80.200,    82.500,
        104.300,   107.600,   110.900,   114.200,   117.500,
        135.300,   139.600,   143.900,   148.200,   152.500,
        166.300,   171.600,   176.900,   182.200,   187.500,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_12:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_12
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
         -44.500,   -46.000,   -47.500,   -49.000,   -50.500,
         -77.500,   -80.000,   -82.500,   -85.000,   -87.500,
        -110.500,  -114.000,  -117.500,  -121.000,  -124.500,
        -143.500,  -148.000,  -152.500,  -157.000,  -161.500,
        -176.500,  -182.000,  -187.500,  -193.000,  -198.500,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_13:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_13
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
         40.100,    41.200,    42.300,    43.400,    44.500,
         69.100,    71.200,    73.300,    75.400,    77.500,
         98.100,   101.200,   104.300,   107.400,   110.500,
        127.100,   131.200,   135.300,   139.400,   143.500,
        156.100,   161.200,   166.300,   171.400,   176.500,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_14:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_14
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
         -40.100,   -41.200,   -42.300,   -43.400,   -44.500,
         -69.100,   -71.200,   -73.300,   -75.400,   -77.500,
         -98.100,  -101.200,  -104.300,  -107.400,  -110.500,
        -127.100,  -131.200,  -135.300,  -139.400,  -143.500,
        -156.100,  -161.200,  -166.300,  -171.400,  -176.500,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_15:
     *      A := A'
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_15
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -113.90,  -116.80,  -119.70,  -122.60,  -125.50,
        -115.00,  -118.00,  -121.00,  -124.00,  -127.00,
        -116.10,  -119.20,  -122.30,  -125.40,  -128.50,
        -117.20,  -120.40,  -123.60,  -126.80,  -130.00,
        -118.30,  -121.60,  -124.90,  -128.20,  -131.50,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_16:
     *      A := A
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_16
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -14.900,   -27.700,   -40.500,   -53.300,   -66.100,
        -25.900,   -48.700,   -71.500,   -94.300,  -117.100,
        -36.900,   -69.700,  -102.500,  -135.300,  -168.100,
        -47.900,   -90.700,  -133.500,  -176.300,  -219.100,
        -58.900,  -111.700,  -164.500,  -217.300,  -270.100,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_17:
     *      A := A'
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_17
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -40.100,   -70.900,  -101.700,  -132.500,  -163.300,
        -39.400,   -71.200,  -103.000,  -134.800,  -166.600,
        -38.700,   -71.500,  -104.300,  -137.100,  -169.900,
        -38.000,   -71.800,  -105.600,  -139.400,  -173.200,
        -37.300,   -72.100,  -106.900,  -141.700,  -176.500,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
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
LM_UT_CASE_FUNC(lm_ut_oper_gemm_3by5)
{
    #undef GEMM_FUNC
    #define GEMM_FUNC(__is_transpose_a, __is_tramspose_b,\
                      __alpha, __mat_a, __mat_b, __beta, __mat_c) \
            lm_oper_gemm(__is_transpose_a, __is_tramspose_b, \
                         __alpha, __mat_a, __mat_b, __beta, __mat_c)

    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_c1_copy = {0};
    lm_mat_t mat_c1_expect = {0};

    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_B_NAME
    #undef ELEM_B_R
    #undef ELEM_B_C
    #undef ELEM_C_NAME
    #undef ELEM_C_R
    #undef ELEM_C_C
    #undef ELEM_C_COPY_NAME
    #undef ELEM_C_COPY_R
    #undef ELEM_C_COPY_C

    #define ELEM_A_NAME         mat_elem_a1
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_B_NAME         mat_elem_b1
    #define ELEM_B_R            3
    #define ELEM_B_C            5
    #define ELEM_C_NAME         mat_elem_c1
    #define ELEM_C_R            3
    #define ELEM_C_C            3
    #define ELEM_C_COPY_NAME    mat_elem_c1_copy
    #define ELEM_C_COPY_R       3
    #define ELEM_C_COPY_C       3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
    };
    lm_mat_elem_t ELEM_B_NAME[ELEM_B_R * ELEM_B_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
    };
    lm_mat_elem_t ELEM_C_NAME[ELEM_C_R * ELEM_C_C] = {
        1.1, 1.2, 1.3,
        2.1, 2.2, 2.3,
        3.1, 3.2, 3.3,
    };
    lm_mat_elem_t ELEM_C_COPY_NAME[ELEM_C_COPY_R * ELEM_C_COPY_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    /*
     * a1_1:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_1
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_2
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_3:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_3
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_4:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_4
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_5:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_5
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_6:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_6
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_7:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_7
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_8:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_8
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_9:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_9
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_10:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_10
    #define ALPHA_MUL           (0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_11:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_11
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_12:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_12
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_13:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_13
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_14:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_14
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_15:
     *      A := A'
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_15
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_16:
     *      A := A
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_16
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -14.900,   -27.700,   -40.500,
        -25.900,   -48.700,   -71.500,
        -36.900,   -69.700,  -102.500,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_17:
     *      A := A'
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_17
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
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
LM_UT_CASE_FUNC(lm_ut_oper_gemm_5by3)
{
    #undef GEMM_FUNC
    #define GEMM_FUNC(__is_transpose_a, __is_tramspose_b,\
                      __alpha, __mat_a, __mat_b, __beta, __mat_c) \
            lm_oper_gemm(__is_transpose_a, __is_tramspose_b, \
                         __alpha, __mat_a, __mat_b, __beta, __mat_c)

    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_c1_copy = {0};
    lm_mat_t mat_c1_expect = {0};

    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_B_NAME
    #undef ELEM_B_R
    #undef ELEM_B_C
    #undef ELEM_C_NAME
    #undef ELEM_C_R
    #undef ELEM_C_C
    #undef ELEM_C_COPY_NAME
    #undef ELEM_C_COPY_R
    #undef ELEM_C_COPY_C

    #define ELEM_A_NAME         mat_elem_a1
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_B_NAME         mat_elem_b1
    #define ELEM_B_R            5
    #define ELEM_B_C            3
    #define ELEM_C_NAME         mat_elem_c1
    #define ELEM_C_R            5
    #define ELEM_C_C            5
    #define ELEM_C_COPY_NAME    mat_elem_c1_copy
    #define ELEM_C_COPY_R       5
    #define ELEM_C_COPY_C       5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.1, 1.2, 1.3,
        2.1, 2.2, 2.3,
        3.1, 3.2, 3.3,
        4.1, 4.2, 4.3,
        5.1, 5.2, 5.3,
    };
    lm_mat_elem_t ELEM_B_NAME[ELEM_B_R * ELEM_B_C] = {
        1.1, 1.2, 1.3,
        2.1, 2.2, 2.3,
        3.1, 3.2, 3.3,
        4.1, 4.2, 4.3,
        5.1, 5.2, 5.3,
    };
    lm_mat_elem_t ELEM_C_NAME[ELEM_C_R * ELEM_C_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_C_COPY_NAME[ELEM_C_COPY_R * ELEM_C_COPY_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    /*
     * a1_1:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_1
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_2
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_3:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_3
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_4:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_4
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_5:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_5
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_6:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_6
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_7:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_7
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_8:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_8
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_9:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_9
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_10:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_10
    #define ALPHA_MUL           (0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_11:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_11
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_12:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_12
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_13:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_13
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_14:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_14
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_15:
     *      A := A'
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_15
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_16:
     *      A := A
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_16
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
         -6.4800,   -13.4800,   -20.4800,   -27.4800,   -34.4800,
        -11.6800,   -24.6800,   -37.6800,   -50.6800,   -63.6800,
        -16.8800,   -35.8800,   -54.8800,   -73.8800,   -92.8800,
        -22.0800,   -47.0800,   -72.0800,   -97.0800,  -122.0800,
        -27.2800,   -58.2800,   -89.2800,  -120.2800,  -151.2800,

    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_17:
     *      A := A'
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_17
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
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
LM_UT_CASE_FUNC(lm_ut_oper_gemm_reshaped_5by1)
{
    #undef GEMM_FUNC
    #define GEMM_FUNC(__is_transpose_a, __is_tramspose_b,\
                      __alpha, __mat_a, __mat_b, __beta, __mat_c) \
            lm_oper_gemm(__is_transpose_a, __is_tramspose_b, \
                         __alpha, __mat_a, __mat_b, __beta, __mat_c)

    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_c1_copy = {0};
    lm_mat_t mat_c1_expect = {0};

    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_B_NAME
    #undef ELEM_B_R
    #undef ELEM_B_C
    #undef ELEM_C_NAME
    #undef ELEM_C_R
    #undef ELEM_C_C
    #undef ELEM_C_COPY_NAME
    #undef ELEM_C_COPY_R
    #undef ELEM_C_COPY_C

    #define ELEM_A_NAME         mat_elem_a1
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_B_NAME         mat_elem_b1
    #define ELEM_B_R            5
    #define ELEM_B_C            5
    #define ELEM_C_NAME         mat_elem_c1
    #define ELEM_C_R            5
    #define ELEM_C_C            5
    #define ELEM_C_COPY_NAME    mat_elem_c1_copy
    #define ELEM_C_COPY_R       5
    #define ELEM_C_COPY_C       5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_B_NAME[ELEM_B_R * ELEM_B_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_C_NAME[ELEM_C_R * ELEM_C_C] = {
        1.1, 1.2, 1.3, 1.4, 1.5,
        2.1, 2.2, 2.3, 2.4, 2.5,
        3.1, 3.2, 3.3, 3.4, 3.5,
        4.1, 4.2, 4.3, 4.4, 4.5,
        5.1, 5.2, 5.3, 5.4, 5.5,
    };
    lm_mat_elem_t ELEM_C_COPY_NAME[ELEM_C_COPY_R * ELEM_C_COPY_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    /*
     * a1_1:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_1
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_2
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_3:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_3
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_4:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_4
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_5:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_5
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_6:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_6
    #define ALPHA_MUL           (0.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_7:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_7
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_8:
     *      A := A
     *      B := B
     *      alpha = 1
     *      beta = -1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_8
    #define ALPHA_MUL           (1.0)
    #define BETA_MUL            (-1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_9:
     *      A := A
     *      B := B
     *      alpha = -1
     *      beta = 1
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_9
    #define ALPHA_MUL           (-1.0)
    #define BETA_MUL            (1.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_10:
     *      A := A
     *      B := B
     *      alpha = 0
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_10
    #define ALPHA_MUL           (0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_11:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = 0
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_11
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (0.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_12:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_12
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_13:
     *      A := A
     *      B := B
     *      alpha = 2
     *      beta = -2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_13
    #define ALPHA_MUL           (2.0)
    #define BETA_MUL            (-2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_14:
     *      A := A
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_14
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_15:
     *      A := A'
     *      B := B
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_15
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      false
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_16:
     *      A := A
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_16
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      false
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        -0.22000,   -2.44000,   -4.66000,   -6.88000,   -9.10000,
        -0.64000,   -5.28000,   -9.92000,  -14.56000,  -19.20000,
        -1.06000,   -8.12000,  -15.18000,  -22.24000,  -29.30000,
        -1.48000,  -10.96000,  -20.44000,  -29.92000,  -39.40000,
        -1.90000,  -13.80000,  -25.70000,  -37.60000,  -49.50000,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_17:
     *      A := A'
     *      B := B'
     *      alpha = -2
     *      beta = 2
     */
    #undef TEST_VAR
    #undef ALPHA_MUL
    #undef BETA_MUL
    #undef IS_TRANSPOSE_A
    #undef IS_TRANSPOSE_B
    #undef ELEM_C_EXPECT_NAME
    #undef ELEM_C_EXPECT_R
    #undef ELEM_C_EXPECT_C

    #define TEST_VAR(var)       var ## 1_17
    #define ALPHA_MUL           (-2.0)
    #define BETA_MUL            (2.0)
    #define IS_TRANSPOSE_A      true
    #define IS_TRANSPOSE_B      true
    #define ELEM_C_EXPECT_NAME  TEST_VAR(mat_elem_c_expect)
    #define ELEM_C_EXPECT_R     ELEM_C_R
    #define ELEM_C_EXPECT_C     ELEM_C_C

    lm_mat_elem_t ELEM_C_EXPECT_NAME[ELEM_C_EXPECT_R * ELEM_C_EXPECT_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_a1, 0, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b1, ELEM_B_R, ELEM_B_C, ELEM_B_NAME,
                        (sizeof(ELEM_B_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c1, ELEM_C_R, ELEM_C_C, ELEM_C_NAME,
                        (sizeof(ELEM_C_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_copy, ELEM_C_COPY_R, ELEM_C_COPY_C, ELEM_C_COPY_NAME,
                        (sizeof(ELEM_C_COPY_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_c1_expect, ELEM_C_EXPECT_R, ELEM_C_EXPECT_C, ELEM_C_EXPECT_NAME,
                        (sizeof(ELEM_C_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    memcpy((void *)(ELEM_C_COPY_NAME), (void *)(ELEM_C_NAME), sizeof(ELEM_C_NAME));

    result = GEMM_FUNC(IS_TRANSPOSE_A, IS_TRANSPOSE_B,
                       ALPHA_MUL, &mat_a1, &mat_b1,
                       BETA_MUL, &mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DOT_PRODUCT_DIM_MISMATCHED)), "");

    result = lm_chk_mat_almost_equal(&mat_c1_copy, &mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_copy);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_c1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

