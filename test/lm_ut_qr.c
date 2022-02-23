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
 * @file    lm_ut_qr.c
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
#include "lm_oper_norm.h"
#include "lm_qr.h"


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
LM_UT_CASE_FUNC(lm_ut_qr_housh_v)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_houshv = {0};
    lm_mat_t mat_a1_expect = {0};
    lm_mat_elem_t alpha1 = {0};
    lm_mat_elem_t beta1 = {0};
    lm_mat_elem_t norm1 = {0};

    /*
     * a2_1:
     *      test 1 by 1 vector, value = 0
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     1
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (0.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 1 by 1 vector, value = -1
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     1
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (0.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
       -1.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
       -1.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_3:
     *      test 1 by 1 vector, value = 1
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 2_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     1
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (0.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_4:
     *      test 1 by 1 vector, value = 10
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 2_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     1
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (0.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        10.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        10.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_5:
     *      test 1 by 1 vector, value = 0.1
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 2_5
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     1
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     1
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (0.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.1,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.1,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 1 by 3 vector, zero vector
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (0.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        0.0, 0.0, 0.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 1 by 3 vector, a(1) = 1, a(2:end) = 0
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (2.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      test 1 by 3 vector, a(1) = 2, a(2:end) = 0
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (2.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        2.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_4:
     *      test 1 by 3 vector, a(1) = -2, a(2:end) = 0
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (2.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
       -2.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, 0.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_5:
     *      test 1 by 3 vector, a(1) = 0, a(2:end) = N
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_5
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (1.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 2.0, 6.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.000000000000000, 0.316227766016837, 0.948683298050513,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_6:
     *      test 1 by 3 vector, a(1) = 0, a(2) = 0, a(3:end) = N
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_6
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (1.0)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, -7.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.0, -1.0,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_7:
     *      test 1 by 3 vector, predefined test vector (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_7
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (1.428571428571429)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        3.0, 2.0, 6.0,
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.0, 0.2, 0.6,
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_8:
     *      test 1 by 3 vector, predefined test vector (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_8
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (1.635083975262450)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.9381746838092822, 0.5546183366882117, 0.9972360891019135
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.000000000000000, 0.229615714353735, 0.412862434995882
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_9:
     *      test 1 by 3 vector, predefined test vector (case 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_HOUSHV_R
    #undef ELEM_A_HOUSHV_C
    #undef ELEM_A_HOUSHV_NAME
    #undef ELEM_A_EXPECT_R
    #undef ELEM_A_EXPECT_C
    #undef BETA_EXPECT

    #define TEST_VAR(var)       var ## 3_9
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_A_HOUSHV_NAME  TEST_VAR(mat_elem_houshv)
    #define ELEM_A_HOUSHV_R     3
    #define ELEM_A_HOUSHV_C     1
    #define ELEM_A_EXPECT_NAME  TEST_VAR(mat_elem_expect)
    #define ELEM_A_EXPECT_R     3
    #define ELEM_A_EXPECT_C     1
    #define BETA_EXPECT         (1.301296155830927)

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.388755015152576, -3.498625927588541, 2.660138409844138
    };
    lm_mat_elem_t ELEM_A_HOUSHV_NAME[ELEM_A_HOUSHV_R * ELEM_A_HOUSHV_C] = {
        0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_A_EXPECT_NAME[ELEM_A_EXPECT_R * ELEM_A_EXPECT_C] = {
        1.000000000000000, 0.583296423825305, -0.443502464526669
    };

    memcpy((void *)ELEM_A_HOUSHV_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_houshv, ELEM_A_HOUSHV_R, ELEM_A_HOUSHV_C, ELEM_A_HOUSHV_NAME,
                        (sizeof(ELEM_A_HOUSHV_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_expect, ELEM_A_EXPECT_R, ELEM_A_EXPECT_C, ELEM_A_EXPECT_NAME,
                        (sizeof(ELEM_A_EXPECT_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_housh_v(&mat_a1_houshv, &alpha1, &beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1_houshv, &mat_a1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The alpha value should equal to +-norm(vector) */
    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_chk_elem_almost_equal(alpha1, (LM_SIGN(ELEM_A_NAME[0]) * norm1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(beta1, BETA_EXPECT);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_houshv);
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
LM_UT_CASE_FUNC(lm_ut_qr_decomp)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_t mat_qr_decomp1 = {0};
    lm_mat_t mat_q1 = {0};
    lm_mat_t mat_q1_mul_r1 = {0};
    lm_mat_t mat_beta1 = {0};
    lm_mat_t mat_qr_work1 = {0};
    lm_mat_t mat_q_work1 = {0};

    /*
     * a1_1:
     *      input Householder beta list matrix with wrong dimension
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         3
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      input Householder beta list matrix with wrong dimension
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      1
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_1:
     *      test 1 by 1 matrix (value = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 1 by 1 matrix (value = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 5 by 5 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 5 by 5 matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,      2.2,    3.3,    0,      5,
        0,      0,      8.8,    0,      10,
        6.3,    0,      0,      0,      6,
        3.2,    3.3,    0,      0,      3,
        5.1,    5.3,    5.5,    1.25,   0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      test 5 by 5 matrix (case 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.20435435140,      2.287,          -3.3087708,     0,              5.0,
        20.54252520,        0,              8.8,            -11.0,          10.0,
        -6.3333333333,      0,              0,              12.504,         6.0,
        35.20210122,        -54.3870,       0,              0,              3.0,
        54.145604,          50.3650678,     50.8707065,     68.278905,      0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      test 3 by 3 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        3, 17,  10,
        2,  4, -2,
        6, 18, -12,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      test 1 by 3 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            3
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  2,  3,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2:
     *      test 3 by 1 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 5_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        2.3054350454,
        3.50468,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);

    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1:
     *      test 3 by 5 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 6_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            5
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,          2.290879087,    3.34505460,     0,      5.546546040,
        0,          0,              8.86048770,     0,      10.506548,
        6.069753,   0,              0,              0,      6.88089801,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);

    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_2:
     *      test 5 by 3 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 6_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,              -2.28768540045,     0,
        0,              5.25547542542,      0,
        0,              8.84726525787687,   0,
        10.12752245,    -6.382768867,       0,
        0,              0,                  -1.12271210,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_1:
     *      test 10 by 10 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 7_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            10
    #define ELEM_A_C            10
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_R
    #define ELEM_QR_DECOMP_C    ELEM_A_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_R
    #define ELEM_Q_MUL_R_C      ELEM_A_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0.0587858,   0.2454289,  -0.0015252,   0.1779677,   0.4629037,  -0.0853936,  -0.1031196,  -0.0916993,   0.1980835,  -0.2531299,
         0.2454289,   0.5430220,   0.1524977,   0.3605385,   0.7865889,  -0.0587622,  -0.0494624,  -0.0609796,   0.3588248,  -0.2651421,
        -0.0015252,   0.1524977,  -0.0523614,   0.1215389,   0.3659989,  -0.0959969,  -0.1230149,  -0.1037855,   0.1492836,  -0.2545600,
         0.1779677,   0.3605385,   0.1215389,   0.2350546,   0.4913275,  -0.0220715,  -0.0095816,  -0.0220164,   0.2279007,  -0.1376540,
         0.4629037,   0.7865889,   0.3659989,   0.4913275,   0.9182210,   0.0360589,   0.0946963,   0.0437703,   0.4458189,  -0.1095322,
        -0.0853936,  -0.0587622,  -0.0959969,  -0.0220715,   0.0360589,  -0.0600301,  -0.0857812,  -0.0657746,   0.0016864,  -0.1217161,
        -0.1031196,  -0.0494624,  -0.1230149,  -0.0095816,   0.0946963,  -0.0857812,  -0.1205962,  -0.0937943,   0.0229330,  -0.1823178,
        -0.0916993,  -0.0609796,  -0.1037855,  -0.0220164,   0.0437703,  -0.0657746,  -0.0937943,  -0.0720495,   0.0038734,  -0.1341916,
         0.1980835,   0.3588248,   0.1492836,   0.2279007,   0.4458189,   0.0016864,   0.0229330,   0.0038734,   0.2123824,  -0.0834124,
        -0.2531299,  -0.2651421,  -0.2545600,  -0.1376540,  -0.1095322,  -0.1217161,  -0.1823178,  -0.1341916,  -0.0834124,  -0.2112969,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_QR_DECOMP_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8_1:
     *      test 3 by 3 sub-matrix of 5 by 5 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_QR_DECOMP_NAME
    #undef ELEM_QR_DECOMP_R
    #undef ELEM_QR_DECOMP_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_R_NAME
    #undef ELEM_R_R
    #undef ELEM_R_C
    #undef ELEM_Q_MUL_R_NAME
    #undef ELEM_Q_MUL_R_R
    #undef ELEM_Q_MUL_R_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_QR_WORK_NAME
    #undef ELEM_QR_WORK_R
    #undef ELEM_QR_WORK_C
    #undef ELEM_Q_WORK_NAME
    #undef ELEM_Q_WORK_R
    #undef ELEM_Q_WORK_C

    #define TEST_VAR(var)       var ## 8_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_A_SHAPED_NAME  TEST_VAR(mat_elem_a_shaped)
    #define ELEM_A_SHAPED_R     3
    #define ELEM_A_SHAPED_C     3
    #define ELEM_QR_DECOMP_NAME TEST_VAR(mat_elem_qr_decomp)
    #define ELEM_QR_DECOMP_R    ELEM_A_SHAPED_R
    #define ELEM_QR_DECOMP_C    ELEM_A_SHAPED_C
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_SHAPED_R
    #define ELEM_Q_C            ELEM_A_SHAPED_R
    #define ELEM_Q_MUL_R_NAME   TEST_VAR(mat_elem_q_mul_r)
    #define ELEM_Q_MUL_R_R      ELEM_A_SHAPED_R
    #define ELEM_Q_MUL_R_C      ELEM_A_SHAPED_C
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_SHAPED_R
    #define ELEM_BETA_C         1
    #define ELEM_QR_WORK_NAME   TEST_VAR(mat_elem_qr_work)
    #define ELEM_QR_WORK_R      1
    #define ELEM_QR_WORK_C      ELEM_A_SHAPED_C
    #define ELEM_Q_WORK_NAME    TEST_VAR(mat_elem_q_work)
    #define ELEM_Q_WORK_R       1
    #define ELEM_Q_WORK_C       ELEM_Q_R

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.20435435140,      2.287,          -3.3087708,     0,              5.0,
        20.54252520,        0,              8.8,            -11.0,          10.0,
        -6.3333333333,      0,              0,              12.504,         6.0,
        35.20210122,        -54.3870,       0,              0,              3.0,
        54.145604,          50.3650678,     50.8707065,     68.278905,      0,
    };
    lm_mat_elem_t ELEM_QR_DECOMP_NAME[ELEM_QR_DECOMP_R * ELEM_QR_DECOMP_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_MUL_R_NAME[ELEM_Q_MUL_R_R * ELEM_Q_MUL_R_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_QR_WORK_NAME[ELEM_QR_WORK_R * ELEM_QR_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_WORK_NAME[ELEM_Q_WORK_R * ELEM_Q_WORK_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_qr_decomp1, ELEM_QR_DECOMP_R, ELEM_QR_DECOMP_C, ELEM_QR_DECOMP_NAME,
                        (sizeof(ELEM_QR_DECOMP_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1_shaped, &mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1_mul_r1, ELEM_Q_MUL_R_R, ELEM_Q_MUL_R_C, ELEM_Q_MUL_R_NAME,
                        (sizeof(ELEM_Q_MUL_R_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_qr_work1, ELEM_QR_WORK_R, ELEM_QR_WORK_C, ELEM_QR_WORK_NAME,
                        (sizeof(ELEM_QR_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q_work1, ELEM_Q_WORK_R, ELEM_Q_WORK_C, ELEM_Q_WORK_NAME,
                        (sizeof(ELEM_Q_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* QR decompose */
    result = lm_qr_decomp(&mat_qr_decomp1, &mat_beta1, &mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_qr_explicit(&mat_qr_decomp1, &mat_beta1, &mat_q1, &mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The Q matrix should be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The R matrix should be a upper triangular matrix */
    result = lm_chk_triu_mat(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q * R must equal to A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_qr_decomp1,
                          LM_MAT_ZERO_VAL, &mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_q1_mul_r1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_decomp1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1_mul_r1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_qr_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_qr_cases[] =
{
    {"lm_ut_qr_housh_v", lm_ut_qr_housh_v, NULL, NULL, 0, 0},
    {"lm_ut_qr_decomp", lm_ut_qr_decomp, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_qr_suites[] =
{
    {"lm_ut_qr_suites", lm_ut_qr_cases, sizeof(lm_ut_qr_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_qr_list[] =
{
    {lm_ut_qr_suites, sizeof(lm_ut_qr_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_qr()
{
    lm_ut_run(lm_qr_list);

    return 0;
}

/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

