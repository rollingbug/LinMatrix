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
 * @file    lm_ut_lu.c
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
#include "lm_oper.h"
#include "lm_oper_dot.h"
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
LM_UT_CASE_FUNC(lm_ut_lu_decomp)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_lu1_decomp = {0};
    lm_mat_t mat_lu1_decomp_shaped = {0};
    lm_mat_t mat_lu1_expect = {0};
    lm_mat_t mat_l1 = {0};
    lm_mat_t mat_u1 = {0};
    lm_mat_t mat_p1 = {0};
    lm_mat_t mat_l1_mul_u1 = {0};
    lm_mat_t mat_p1_mul_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_permute_list_t perm_list1 = {0};
    int32_t inv_sgn_det1 = {0};
    lm_mat_dim_size_t max_pivot_r1;
    lm_mat_dim_size_t max_pivot_c1;
    lm_mat_elem_t max_pivot_abs1;

    /*
     * a1: Test the 2 by 2 square matrix (to verify the partial pivoting)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a1
    #define ELEM_A_R                2
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu1_decomp
    #define ELEM_LU_DECOMP_R        2
    #define ELEM_LU_DECOMP_C        2
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu1_expect
    #define ELEM_LU_EXPECT_R        2
    #define ELEM_LU_EXPECT_C        2
    #define ELEM_L_NAME             mat_elem_l1
    #define ELEM_L_R                2
    #define ELEM_L_C                2
    #define ELEM_U_NAME             mat_elem_u1
    #define ELEM_U_R                2
    #define ELEM_U_C                2
    #define ELEM_P_NAME             mat_elem_p1
    #define ELEM_P_R                2
    #define ELEM_P_C                2
    #define ELEM_L_MUL_U_NAME       mat_elem_l1_mul_u1
    #define ELEM_L_MUL_U_R          2
    #define ELEM_L_MUL_U_C          2
    #define ELEM_P_MUL_A_NAME       mat_elem_p1_mul_a1
    #define ELEM_P_MUL_A_R          2
    #define ELEM_P_MUL_A_C          2
    #define ELEM_PERM_LIST          perm_elem_list1
    #define ELEM_PERM_LIST_SIZE     4

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0001, 1,
        1,      1,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        1,      1,
        0.0001, 0.9999,

    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0,
        0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test the 3 by 3 square matrix (to verify the partial pivoting)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a2
    #define ELEM_A_R                3
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu2_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu2_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_L_NAME             mat_elem_l2
    #define ELEM_L_R                3
    #define ELEM_L_C                3
    #define ELEM_U_NAME             mat_elem_u2
    #define ELEM_U_R                3
    #define ELEM_U_C                3
    #define ELEM_P_NAME             mat_elem_p2
    #define ELEM_P_R                3
    #define ELEM_P_C                3
    #define ELEM_L_MUL_U_NAME       mat_elem_l2_mul_u2
    #define ELEM_L_MUL_U_R          3
    #define ELEM_L_MUL_U_C          3
    #define ELEM_P_MUL_A_NAME       mat_elem_p2_mul_a2
    #define ELEM_P_MUL_A_R          3
    #define ELEM_P_MUL_A_C          3
    #define ELEM_PERM_LIST          perm_elem_list2
    #define ELEM_PERM_LIST_SIZE     6

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        3,  17, 10,
        2,  4,  -2,
        6,  18, -12
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
         6.000000000000000, 18.000000000000000, -12.000000000000000,
         0.500000000000000,  8.000000000000000,  16.000000000000000,
         0.333333333333333, -0.250000000000000,   6.000000000000000,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * I disabled this test checkpoint because I found that the LU decomposition
     * function may generate different L, U, and P result on different CPUs/platforms
     * due to floating point calculation errors. Verifying the value of the L matrix
     * directly does not seem to be the correct method for For testing.
     */
//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test the 6 by 6 square ZERO matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a3
    #define ELEM_A_R                6
    #define ELEM_A_C                6
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu3_decomp
    #define ELEM_LU_DECOMP_R        6
    #define ELEM_LU_DECOMP_C        6
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu3_expect
    #define ELEM_LU_EXPECT_R        6
    #define ELEM_LU_EXPECT_C        6
    #define ELEM_L_NAME             mat_elem_l3
    #define ELEM_L_R                6
    #define ELEM_L_C                6
    #define ELEM_U_NAME             mat_elem_u3
    #define ELEM_U_R                6
    #define ELEM_U_C                6
    #define ELEM_P_NAME             mat_elem_p3
    #define ELEM_P_R                6
    #define ELEM_P_C                6
    #define ELEM_L_MUL_U_NAME       mat_elem_l3_mul_u3
    #define ELEM_L_MUL_U_R          6
    #define ELEM_L_MUL_U_C          6
    #define ELEM_P_MUL_A_NAME       mat_elem_p3_mul_a3
    #define ELEM_P_MUL_A_R          6
    #define ELEM_P_MUL_A_C          6
    #define ELEM_PERM_LIST          perm_elem_list3
    #define ELEM_PERM_LIST_SIZE     12

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test the 6 by 2 non-square ZERO matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a4
    #define ELEM_A_R                6
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu4_decomp
    #define ELEM_LU_DECOMP_R        6
    #define ELEM_LU_DECOMP_C        2
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu4_expect
    #define ELEM_LU_EXPECT_R        6
    #define ELEM_LU_EXPECT_C        2
    #define ELEM_L_NAME             mat_elem_l4
    #define ELEM_L_R                6
    #define ELEM_L_C                2
    #define ELEM_U_NAME             mat_elem_u4
    #define ELEM_U_R                2
    #define ELEM_U_C                2
    #define ELEM_P_NAME             mat_elem_p4
    #define ELEM_P_R                6
    #define ELEM_P_C                6
    #define ELEM_L_MUL_U_NAME       mat_elem_l4_mul_u4
    #define ELEM_L_MUL_U_R          6
    #define ELEM_L_MUL_U_C          2
    #define ELEM_P_MUL_A_NAME       mat_elem_p4_mul_a4
    #define ELEM_P_MUL_A_R          6
    #define ELEM_P_MUL_A_C          2
    #define ELEM_PERM_LIST          perm_elem_list4
    #define ELEM_PERM_LIST_SIZE     12

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    lm_shape_submatrix(&mat_lu1_decomp, 0, 0, ELEM_U_R, ELEM_U_C, &mat_lu1_decomp_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp_shaped, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test the 1 by 1 square identity matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a5
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu5_decomp
    #define ELEM_LU_DECOMP_R        1
    #define ELEM_LU_DECOMP_C        1
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu5_expect
    #define ELEM_LU_EXPECT_R        1
    #define ELEM_LU_EXPECT_C        1
    #define ELEM_L_NAME             mat_elem_l5
    #define ELEM_L_R                1
    #define ELEM_L_C                1
    #define ELEM_U_NAME             mat_elem_u5
    #define ELEM_U_R                1
    #define ELEM_U_C                1
    #define ELEM_P_NAME             mat_elem_p5
    #define ELEM_P_R                1
    #define ELEM_P_C                1
    #define ELEM_L_MUL_U_NAME       mat_elem_l5_mul_u5
    #define ELEM_L_MUL_U_R          1
    #define ELEM_L_MUL_U_C          1
    #define ELEM_P_MUL_A_NAME       mat_elem_p5_mul_a5
    #define ELEM_P_MUL_A_R          1
    #define ELEM_P_MUL_A_C          1
    #define ELEM_PERM_LIST          perm_elem_list5
    #define ELEM_PERM_LIST_SIZE     2

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    //result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test the 6 by 6 square identity matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a6
    #define ELEM_A_R                6
    #define ELEM_A_C                6
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu6_decomp
    #define ELEM_LU_DECOMP_R        6
    #define ELEM_LU_DECOMP_C        6
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu6_expect
    #define ELEM_LU_EXPECT_R        6
    #define ELEM_LU_EXPECT_C        6
    #define ELEM_L_NAME             mat_elem_l6
    #define ELEM_L_R                6
    #define ELEM_L_C                6
    #define ELEM_U_NAME             mat_elem_u6
    #define ELEM_U_R                6
    #define ELEM_U_C                6
    #define ELEM_P_NAME             mat_elem_p6
    #define ELEM_P_R                6
    #define ELEM_P_C                6
    #define ELEM_L_MUL_U_NAME       mat_elem_l6_mul_u6
    #define ELEM_L_MUL_U_R          6
    #define ELEM_L_MUL_U_C          6
    #define ELEM_P_MUL_A_NAME       mat_elem_p6_mul_a6
    #define ELEM_P_MUL_A_R          6
    #define ELEM_P_MUL_A_C          6
    #define ELEM_PERM_LIST          perm_elem_list6
    #define ELEM_PERM_LIST_SIZE     12

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test the 5 * 5 predefined square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a7
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu7_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu7_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l7
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u7
    #define ELEM_U_R                5
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p7
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l7_mul_u7
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p7_mul_a7
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list7
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        1,  2,  3,  4,  5,
        1,  0,  0,  0,  0,
        1,  0,  0,  0,  0,
        1,  0,  0,  0,  0,
        1,  0,  0,  0,  0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test the 5 * 5 predefined square matrix (case 2)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a8
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu8_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu8_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l8
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u8
    #define ELEM_U_R                5
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p8
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l8_mul_u8
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p8_mul_a8
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list8
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  1,  1,  1,  1,
        2,  2,  2,  2,  2,
        3,  3,  3,  3,  3,
        4,  4,  4,  4,  4,
        5,  5,  5,  5,  5,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        5,      5,  5,  5,  5,
        0.4,    0,  0,  0,  0,
        0.6,    0,  0,  0,  0,
        0.8,    0,  0,  0,  0,
        0.2,    0,  0,  0,  0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a9: Test the 5 * 5 predefined square matrix (case 3)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a9
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu9_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu9_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l9
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u9
    #define ELEM_U_R                5
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p9
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l9_mul_u9
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p9_mul_a9
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list9
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,      2.2,    3.3,    0,      5,
        0,      0,      8.8,    0,      10,
        6.3,    0,      0,      0,      6,
        3.2,    3.3,    0,      0,      3,
        5.1,    5.3,    5.5,    1.25,   0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        6.300000000000000,   0.000000000000000,   0.000000000000000,   0.000000000000000,   6.000000000000000,
        0.809523809523809,   5.300000000000000,   5.500000000000000,   1.250000000000000,  -4.857142857142857,
        0.000000000000000,   0.000000000000000,   8.800000000000001,   0.000000000000000,  10.000000000000000,
        0.507936507936507,   0.622641509433962,  -0.389150943396226,  -0.778301886792452,   6.868149146451033,
        0.000000000000000,   0.415094339622641,   0.115566037735849,   0.666666666666666,   1.281746031746032,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a10: Test the 5 * 5 predefined square matrix (case 4)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a10
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu10_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu10_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l10
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u10
    #define ELEM_U_R                5
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p10
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l10_mul_u10
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p10_mul_a10
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list10
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,  0,  0,      0,      5,
        0,  0,  8.8,    0,      10,
        0,  0,  0,      1,      10,
        0,  0,  0,     -3,      10,
        0,  0,  5.5,    5,      10,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
         0.000000000000000e+00,   0.000000000000000e+00,   8.800000000000001e+00,   0.000000000000000e+00,   1.000000000000000e+01,
         6.250000000000000e-01,   0.000000000000000e+00,   0.000000000000000e+00,   5.000000000000000e+00,   3.750000000000000e+00,
         0.000000000000000e+00,  -6.000000000000000e-01,   0.000000000000000e+00,   0.000000000000000e+00,   1.225000000000000e+01,
         0.000000000000000e+00,   2.000000000000000e-01,   7.551020408163265e-01,   0.000000000000000e+00,   0.000000000000000e+00,
         0.000000000000000e+00,   0.000000000000000e+00,   4.081632653061225e-01,   0.000000000000000e+00,   0.000000000000000e+00,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a11: Test the 5 * 5 predefined square matrix (case 5)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a11
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu11_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu11_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l11
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u11
    #define ELEM_U_R                5
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p11
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l11_mul_u11
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p11_mul_a11
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list11
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         8.321103886879729e-32,   2.919265825302702e-31,  -5.227931840117220e-32,   1.919858442405203e-31,  -2.412146491955093e-31,
        -7.292259031654114e-32,   1.025738361687426e-31,   6.597461091531938e-31,   1.410964464050669e-32,  -2.055094259356599e-31,
         3.634016798949415e-31,  -2.404682295529631e-31,  -4.596309461611641e-32,  -5.752566087876835e-32,  -1.093207043154580e-31,
        -2.086385964117051e-31,  -3.039824689477334e-31,   1.197644944396642e-31,  -4.049306121491250e-32,  -1.678020481232411e-31,
        -5.651622488122478e-31,   7.521804777038970e-32,   3.037317689091766e-31,   3.428182281184684e-31,  -8.835888107160591e-31,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        -5.651622488122478e-31,   7.521804777038970e-32,   3.037317689091766e-31,   3.428182281184684e-31,  -8.835888107160591e-31,
         3.691658401639222e-01,  -3.317504027483793e-31,   7.637100785833751e-33,  -1.670498404217740e-31,   1.583887575441937e-31,
         1.290294786493546e-01,  -2.799348242248988e-01,   6.226936478276667e-31,  -7.688708032038150e-32,  -4.716189334622955e-32,
        -6.430041650139781e-01,   5.790579603247238e-01,   2.327233854741585e-01,   2.775328491427047e-31,  -7.582125851626337e-31,
        -1.472338944147007e-01,  -9.133409518830069e-01,  -9.385518992962319e-04,   3.236182114736778e-01,   1.868120456708269e-32,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a11: Test the 1 * 3 predefined non-square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a12
    #define ELEM_A_R                1
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu12_decomp
    #define ELEM_LU_DECOMP_R        1
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu12_expect
    #define ELEM_LU_EXPECT_R        1
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_L_NAME             mat_elem_l12
    #define ELEM_L_R                1
    #define ELEM_L_C                1
    #define ELEM_U_NAME             mat_elem_u12
    #define ELEM_U_R                1
    #define ELEM_U_C                3
    #define ELEM_P_NAME             mat_elem_p12
    #define ELEM_P_R                1
    #define ELEM_P_C                1
    #define ELEM_L_MUL_U_NAME       mat_elem_l12_mul_u12
    #define ELEM_L_MUL_U_R          1
    #define ELEM_L_MUL_U_C          3
    #define ELEM_P_MUL_A_NAME       mat_elem_p12_mul_a12
    #define ELEM_P_MUL_A_R          1
    #define ELEM_P_MUL_A_C          3
    #define ELEM_PERM_LIST          perm_elem_list12
    #define ELEM_PERM_LIST_SIZE     2

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  2,  3,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        1,  2,  3,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    //result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a13: Test the 3 * 1 predefined non-square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a13
    #define ELEM_A_R                3
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu13_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        1
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu13_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        1
    #define ELEM_L_NAME             mat_elem_l13
    #define ELEM_L_R                3
    #define ELEM_L_C                1
    #define ELEM_U_NAME             mat_elem_u13
    #define ELEM_U_R                1
    #define ELEM_U_C                1
    #define ELEM_P_NAME             mat_elem_p13
    #define ELEM_P_R                3
    #define ELEM_P_C                3
    #define ELEM_L_MUL_U_NAME       mat_elem_l13_mul_u13
    #define ELEM_L_MUL_U_R          3
    #define ELEM_L_MUL_U_C          1
    #define ELEM_P_MUL_A_NAME       mat_elem_p13_mul_a13
    #define ELEM_P_MUL_A_R          3
    #define ELEM_P_MUL_A_C          1
    #define ELEM_PERM_LIST          perm_elem_list13
    #define ELEM_PERM_LIST_SIZE     6

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        2.3054350454,
        3.50468,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
        0,
        0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {

        3.504680000000000,
        0.657816133113436,
        0.000000000000000,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0,
        0,
        0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0,
        0,
        0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0,
        0,
        0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    lm_shape_submatrix(&mat_lu1_decomp, 0, 0, ELEM_U_R, ELEM_U_C, &mat_lu1_decomp_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp_shaped, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a14: Test the 3 * 5 predefined non-square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a14
    #define ELEM_A_R                3
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu14_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu14_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l14
    #define ELEM_L_R                3
    #define ELEM_L_C                3
    #define ELEM_U_NAME             mat_elem_u14
    #define ELEM_U_R                3
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p14
    #define ELEM_P_R                3
    #define ELEM_P_C                3
    #define ELEM_L_MUL_U_NAME       mat_elem_l14_mul_u14
    #define ELEM_L_MUL_U_R          3
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p14_mul_a14
    #define ELEM_P_MUL_A_R          3
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list14
    #define ELEM_PERM_LIST_SIZE     6

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
        1,  2,  3,  4,  5,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        1,  2,  3,  4,  5,
        1,  0,  0,  0,  0,
        1,  0,  0,  0,  0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    lm_shape_submatrix(&mat_lu1_decomp, 0, 0, ELEM_L_R, ELEM_L_C, &mat_lu1_decomp_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp_shaped, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a15: Test the 3 * 5 predefined non-square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a15
    #define ELEM_A_R                3
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu15_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu15_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l15
    #define ELEM_L_R                3
    #define ELEM_L_C                3
    #define ELEM_U_NAME             mat_elem_u15
    #define ELEM_U_R                3
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p15
    #define ELEM_P_R                3
    #define ELEM_P_C                3
    #define ELEM_L_MUL_U_NAME       mat_elem_l15_mul_u15
    #define ELEM_L_MUL_U_R          3
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p15_mul_a15
    #define ELEM_P_MUL_A_R          3
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list15
    #define ELEM_PERM_LIST_SIZE     6

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 1, 0, 0,
        0, 1, 2, 6, 2,
        0, 2, 3, 8, 7,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        0.000000000000000e+00,   2.000000000000000e+00,   3.000000000000000e+00,   8.000000000000000e+00,   7.000000000000000e+00,
        0.000000000000000e+00,   0.000000000000000e+00,   1.000000000000000e+00,   0.000000000000000e+00,   0.000000000000000e+00,
        5.000000000000000e-01,   5.000000000000000e-01,   0.000000000000000e+00,   2.000000000000000e+00,  -1.500000000000000e+00,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    lm_shape_submatrix(&mat_lu1_decomp, 0, 0, ELEM_L_R, ELEM_L_C, &mat_lu1_decomp_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp_shaped, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a16: Test the 5 * 3 predefined non-square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a16
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu16_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu16_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_L_NAME             mat_elem_l16
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u16
    #define ELEM_U_R                5
    #define ELEM_U_C                3
    #define ELEM_P_NAME             mat_elem_p16
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l16_mul_u16
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          3
    #define ELEM_P_MUL_A_NAME       mat_elem_p16_mul_a16
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          3
    #define ELEM_PERM_LIST          perm_elem_list16
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,              -2.28768540045,     0,
        0,              5.25547542542,      0,
        0,              8.84726525787687,   0,
        10.12752245,    -6.382768867,       0,
        0,              0,                  -1.12271210,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
       10.127522450000000,  -6.382768867000000,   0.000000000000000,
        0.000000000000000,   8.847265257876870,   0.000000000000000,
        0.000000000000000,   0.000000000000000,  -1.122712100000000,
        0.000000000000000,  -0.258575427973433,  -0.000000000000000,
        0.000000000000000,   0.594022590284716,  -0.000000000000000,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    lm_shape_submatrix(&mat_l1, 0, 0, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C, &mat_lu1_decomp_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_lu1_decomp_shaped, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a17: Test the 5 * 3 predefined non-square matrix (case 2)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a17
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu17_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu17_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_L_NAME             mat_elem_l17
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u17
    #define ELEM_U_R                5
    #define ELEM_U_C                3
    #define ELEM_P_NAME             mat_elem_p17
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l17_mul_u17
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          3
    #define ELEM_P_MUL_A_NAME       mat_elem_p17_mul_a17
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          3
    #define ELEM_PERM_LIST          perm_elem_list17
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0,
        0, 0, 1,
        0, 0, 7,
        0, 0, 9,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        0,                      0, 9,
        1.111111111111111e-01,  0, 0,
        7.777777777777778e-01,  0, 0,
        0,                      0, 0,
        0,                      0, 0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    lm_shape_submatrix(&mat_l1, 0, 0, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C, &mat_lu1_decomp_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_lu1_decomp_shaped, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a18: Test the 1 * 1 predefined square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a18
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu18_decomp
    #define ELEM_LU_DECOMP_R        1
    #define ELEM_LU_DECOMP_C        1
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu18_expect
    #define ELEM_LU_EXPECT_R        1
    #define ELEM_LU_EXPECT_C        1
    #define ELEM_L_NAME             mat_elem_l18
    #define ELEM_L_R                1
    #define ELEM_L_C                1
    #define ELEM_U_NAME             mat_elem_u18
    #define ELEM_U_R                1
    #define ELEM_U_C                1
    #define ELEM_P_NAME             mat_elem_p18
    #define ELEM_P_R                1
    #define ELEM_P_C                1
    #define ELEM_L_MUL_U_NAME       mat_elem_l18_mul_u18
    #define ELEM_L_MUL_U_R          1
    #define ELEM_L_MUL_U_C          1
    #define ELEM_P_MUL_A_NAME       mat_elem_p18_mul_a18
    #define ELEM_P_MUL_A_R          1
    #define ELEM_P_MUL_A_C          1
    #define ELEM_PERM_LIST          perm_elem_list18
    #define ELEM_PERM_LIST_SIZE     2

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.7,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        -0.7,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    //lm_shape_submatrix(&mat_l1, 0, 0, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C, &mat_lu1_decomp_shaped);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    //result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_lu1_decomp_shaped, -1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a19: Test the 3 by 3 sub-matrix of predefined non-square matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a19
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_A_SHAPED_NAME      mat_elem_a19_shaped
    #define ELEM_A_SHAPED_R         3
    #define ELEM_A_SHAPED_C         3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu19_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu19_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_L_NAME             mat_elem_l19
    #define ELEM_L_R                3
    #define ELEM_L_C                3
    #define ELEM_U_NAME             mat_elem_u19
    #define ELEM_U_R                3
    #define ELEM_U_C                3
    #define ELEM_P_NAME             mat_elem_p19
    #define ELEM_P_R                3
    #define ELEM_P_C                3
    #define ELEM_L_MUL_U_NAME       mat_elem_l19_mul_u19
    #define ELEM_L_MUL_U_R          3
    #define ELEM_L_MUL_U_C          3
    #define ELEM_P_MUL_A_NAME       mat_elem_p19_mul_a19
    #define ELEM_P_MUL_A_R          3
    #define ELEM_P_MUL_A_C          3
    #define ELEM_PERM_LIST          perm_elem_list19
    #define ELEM_PERM_LIST_SIZE     6

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,              -2.28768540045,     0,
        0,              5.25547542542,      0,
        0,              8.84726525787687,   0,
        10.12752245,    -6.382768867,       0,
        0,              0,                  -1.12271210,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        10.12752245000000, -6.382768867000000, 0.000000000000000,
        0.000000000000000,  8.847265257876870, 0.000000000000000,
        0.000000000000000,  0.594022590284716, 0.000000000000000,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 0,
                                ELEM_A_SHAPED_R, ELEM_A_SHAPED_R,
                                &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1_shaped, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1_shaped, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a20: Test 5 by 5 predefined square matrix
     * (only one element is not equal to zero)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_L_NAME
    #undef ELEM_L_R
    #undef ELEM_L_C
    #undef ELEM_U_NAME
    #undef ELEM_U_R
    #undef ELEM_U_C
    #undef ELEM_P_NAME
    #undef ELEM_P_R
    #undef ELEM_P_C
    #undef ELEM_L_MUL_U_NAME
    #undef ELEM_L_MUL_U_R
    #undef ELEM_L_MUL_U_C
    #undef ELEM_P_MUL_A_NAME
    #undef ELEM_P_MUL_A_R
    #undef ELEM_P_MUL_A_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE

    #define ELEM_A_NAME             mat_elem_a20
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu20_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu20_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_L_NAME             mat_elem_l20
    #define ELEM_L_R                5
    #define ELEM_L_C                5
    #define ELEM_U_NAME             mat_elem_u20
    #define ELEM_U_R                5
    #define ELEM_U_C                5
    #define ELEM_P_NAME             mat_elem_p20
    #define ELEM_P_R                5
    #define ELEM_P_C                5
    #define ELEM_L_MUL_U_NAME       mat_elem_l20_mul_u20
    #define ELEM_L_MUL_U_R          5
    #define ELEM_L_MUL_U_C          5
    #define ELEM_P_MUL_A_NAME       mat_elem_p20_mul_a20
    #define ELEM_P_MUL_A_R          5
    #define ELEM_P_MUL_A_C          5
    #define ELEM_PERM_LIST          perm_elem_list20
    #define ELEM_PERM_LIST_SIZE     10

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 5, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        0, 0, 5, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_L_NAME[ELEM_L_R * ELEM_L_C] = {
        0
    };
    lm_mat_elem_t ELEM_U_NAME[ELEM_U_R * ELEM_U_C] = {
        0
    };
    lm_mat_elem_t ELEM_P_NAME[ELEM_P_R * ELEM_P_C] = {
        0
    };
    lm_mat_elem_t ELEM_L_MUL_U_NAME[ELEM_L_MUL_U_R * ELEM_L_MUL_U_C] = {
        0
    };
    lm_mat_elem_t ELEM_P_MUL_A_NAME[ELEM_P_MUL_A_R * ELEM_P_MUL_A_C] = {
        0
    };
    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1, ELEM_L_R, ELEM_L_C,
                        ELEM_L_NAME, sizeof(ELEM_L_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_u1, ELEM_U_R, ELEM_U_C,
                        ELEM_U_NAME, sizeof(ELEM_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1, ELEM_P_R, ELEM_P_C,
                        ELEM_P_NAME, sizeof(ELEM_P_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_l1_mul_u1, ELEM_L_MUL_U_R, ELEM_L_MUL_U_C,
                        ELEM_L_MUL_U_NAME, sizeof(ELEM_L_MUL_U_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_p1_mul_a1, ELEM_P_MUL_A_R, ELEM_P_MUL_A_C,
                        ELEM_P_MUL_A_NAME, sizeof(ELEM_P_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Prepare L, U, P matrix */
    result = lm_oper_identity(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_tril(&mat_lu1_decomp, &mat_l1, -1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_zeros(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_copy_triu(&mat_lu1_decomp, &mat_u1, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_identity(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_oper_permute_row(&mat_p1, &perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Validate: all the absolute value in matrix L should
     *           be less than or equal to 1 if the partial
     *           pivot function works well.
     */
    result = lm_oper_max_abs(&mat_l1, &max_pivot_r1, &max_pivot_c1, &max_pivot_abs1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((max_pivot_abs1 <= LM_MAT_ONE_VAL), "");

    /* Calculate PA */
    result = lm_oper_dot(&mat_p1, &mat_a1, &mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate LU */
    result = lm_oper_dot(&mat_l1, &mat_u1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check PA == LU */
    result = lm_chk_mat_almost_equal(&mat_p1_mul_a1, &mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_l1_mul_u1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_p1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
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
LM_UT_CASE_FUNC(lm_ut_lu_det)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_lu1_decomp = {0};
    lm_mat_t mat_lu1_expect = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_permute_list_t perm_list1 = {0};
    lm_mat_elem_t det1 = {0};
    int32_t inv_sgn_det1 = {0};

    /*
     * a2: Test 1 by 1 square matrix (det = 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a2
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu2_decomp
    #define ELEM_LU_DECOMP_R        1
    #define ELEM_LU_DECOMP_C        1
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu2_expect
    #define ELEM_LU_EXPECT_R        1
    #define ELEM_LU_EXPECT_C        1
    #define ELEM_PERM_LIST          perm_elem_list2
    #define ELEM_PERM_LIST_SIZE     2
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(LM_MAT_ONE_VAL))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        LM_MAT_ONE_VAL
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        LM_MAT_ONE_VAL,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /*
     * a3: Test 1 by 1 square matrix (det = 0)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a3
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu3_decomp
    #define ELEM_LU_DECOMP_R        1
    #define ELEM_LU_DECOMP_C        1
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu3_expect
    #define ELEM_LU_EXPECT_R        1
    #define ELEM_LU_EXPECT_C        1
    #define ELEM_PERM_LIST          perm_elem_list3
    #define ELEM_PERM_LIST_SIZE     2
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(LM_MAT_ZERO_VAL))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        LM_MAT_ZERO_VAL
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        LM_MAT_ZERO_VAL,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 2 by 2 square matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a4
    #define ELEM_A_R                2
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu4_decomp
    #define ELEM_LU_DECOMP_R        2
    #define ELEM_LU_DECOMP_C        2
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu4_expect
    #define ELEM_LU_EXPECT_R        2
    #define ELEM_LU_EXPECT_C        2
    #define ELEM_PERM_LIST          perm_elem_list4
    #define ELEM_PERM_LIST_SIZE     4
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(-0.9999))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0001, 1,
        1,      1,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        1,      1,
        0.0001, 0.9999,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 3 by 3 square matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a5
    #define ELEM_A_R                3
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu5_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu5_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_PERM_LIST          perm_elem_list5
    #define ELEM_PERM_LIST_SIZE     6
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(288.0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        3,  17, 10,
        2,  4,  -2,
        6,  18, -12
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
         6.000000000000000, 18.000000000000000, -12.000000000000000,
         0.500000000000000,  8.000000000000000,  16.000000000000000,
         0.333333333333333, -0.250000000000000,   6.000000000000000,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 6 by 6 square identity matrix
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a6
    #define ELEM_A_R                6
    #define ELEM_A_C                6
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu6_decomp
    #define ELEM_LU_DECOMP_R        6
    #define ELEM_LU_DECOMP_C        6
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu6_expect
    #define ELEM_LU_EXPECT_R        6
    #define ELEM_LU_EXPECT_C        6
    #define ELEM_PERM_LIST          perm_elem_list6
    #define ELEM_PERM_LIST_SIZE     12
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(LM_MAT_ONE_VAL))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,  LM_MAT_ZERO_VAL,
        LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test 5 by 5 square identity matrix (det = 0)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a7
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu7_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu7_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_PERM_LIST          perm_elem_list7
    #define ELEM_PERM_LIST_SIZE     10
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(LM_MAT_ZERO_VAL))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  1,  1,  1,  1,
        2,  2,  2,  2,  2,
        3,  3,  3,  3,  3,
        4,  4,  4,  4,  4,
        5,  5,  5,  5,  5,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        5,      5,  5,  5,  5,
        0.4,    0,  0,  0,  0,
        0.6,    0,  0,  0,  0,
        0.8,    0,  0,  0,  0,
        0.2,    0,  0,  0,  0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a8: Test 5 by 5 predefined matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a8
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu8_decomp
    #define ELEM_LU_DECOMP_R        5
    #define ELEM_LU_DECOMP_C        5
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu8_expect
    #define ELEM_LU_EXPECT_R        5
    #define ELEM_LU_EXPECT_C        5
    #define ELEM_PERM_LIST          perm_elem_list8
    #define ELEM_PERM_LIST_SIZE     10
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(293.1225000000000023))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,      2.2,    3.3,    0,      5,
        0,      0,      8.8,    0,      10,
        6.3,    0,      0,      0,      6,
        3.2,    3.3,    0,      0,      3,
        5.1,    5.3,    5.5,    1.25,   0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        6.300000000000000,   0.000000000000000,   0.000000000000000,   0.000000000000000,   6.000000000000000,
        0.809523809523809,   5.300000000000000,   5.500000000000000,   1.250000000000000,  -4.857142857142857,
        0.000000000000000,   0.000000000000000,   8.800000000000001,   0.000000000000000,  10.000000000000000,
        0.507936507936508,   0.622641509433962,  -0.389150943396226,  -0.778301886792453,   6.868149146451033,
        0.000000000000000,   0.415094339622642,   0.115566037735849,   0.666666666666667,   1.281746031746032,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a9: Test 1 by 3 predefined matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a9
    #define ELEM_A_R                1
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu9_decomp
    #define ELEM_LU_DECOMP_R        1
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu9_expect
    #define ELEM_LU_EXPECT_R        1
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_PERM_LIST          perm_elem_list9
    #define ELEM_PERM_LIST_SIZE     2
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,  2,  3,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        1,  2,  3,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a10: Test 3 by 1 predefined matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a10
    #define ELEM_A_R                3
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu10_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        1
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu10_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        1
    #define ELEM_PERM_LIST          perm_elem_list10
    #define ELEM_PERM_LIST_SIZE     6
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        2.3054350454,
        3.50468,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
        0,
        0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {

        3.504680000000000,
        0.657816133113436,
        0.000000000000000,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a11: Test 3 by 1 predefined matrix (case 1)
     */
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_LU_EXPECT_NAME
    #undef ELEM_LU_EXPECT_R
    #undef ELEM_LU_EXPECT_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef DETERMINANT_EXPECT

    #define ELEM_A_NAME             mat_elem_a11
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     mat_elem_lu11_decomp
    #define ELEM_LU_DECOMP_R        3
    #define ELEM_LU_DECOMP_C        3
    #define ELEM_LU_EXPECT_NAME     mat_elem_lu11_expect
    #define ELEM_LU_EXPECT_R        3
    #define ELEM_LU_EXPECT_C        3
    #define ELEM_PERM_LIST          perm_elem_list11
    #define ELEM_PERM_LIST_SIZE     6
    #define DETERMINANT_EXPECT      ((lm_mat_elem_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,              -2.28768540045,     0,
        0,              5.25547542542,      0,
        0,              8.84726525787687,   0,
        10.12752245,    -6.382768867,       0,
        0,              0,                  -1.12271210,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_EXPECT_NAME[ELEM_LU_EXPECT_R * ELEM_LU_EXPECT_C] = {
        10.12752245000000, -6.382768867000000, 0.000000000000000,
        0.000000000000000,  8.847265257876870, 0.000000000000000,
        0.000000000000000,  0.594022590284716, 0.000000000000000,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 0,
                                ELEM_A_SHAPED_R, ELEM_A_SHAPED_R,
                                &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_expect, ELEM_LU_EXPECT_R, ELEM_LU_EXPECT_C,
                        ELEM_LU_EXPECT_NAME, sizeof(ELEM_LU_EXPECT_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1_shaped, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

//    result = lm_chk_mat_almost_equal(&mat_lu1_decomp, &mat_lu1_expect);
//    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(det1, DETERMINANT_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong determinant");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_expect);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
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
LM_UT_CASE_FUNC(lm_ut_lu_rank)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_lu1_decomp = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_permute_list_t perm_list1 = {0};
    int32_t inv_sgn_det1 = {0};
    lm_mat_elem_size_t rank1 = {0};

    /*
     * a2_1: Test 1 by 1 square matrix (rank = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 2_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        LM_MAT_ZERO_VAL
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_1: Test 1 by 1 square matrix (rank = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 2_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(1))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        100,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1: Test 1 by 5 square matrix (rank = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 3_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,
        0,
        0,
        0,
        0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2: Test 1 by 5 square matrix (rank = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 3_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(1))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        100,
        5,
        10,
        7,
        0.8,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1: Test 5 by 1 square matrix (rank = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 4_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2: Test 5 by 1 square matrix (rank = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 4_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(1))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0, 0.1, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1: Test 5 by 3 square matrix (rank = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(1))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0,
        0, 0, 1,
        0, 0, 7,
        0, 0, 9,
        0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2: Test 5 by 3 square matrix (rank = 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(3))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 0, 0,
        2, 6, 0,
        3, 8, 7,
        4, 0, 9,
        5, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_3: Test 5 by 3 square matrix (rank = 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(3))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -5.698181529766688e-02,   2.250483877966437e-01,  -1.383849922723435e-02,
         9.439994468422908e-02,   1.212984400360769e-01,  -4.507837121025215e-01,
        -6.513553282258426e-01,   1.465575091649813e-01,   2.335760150884632e-01,
        -7.241582273531633e-01,  -2.524736273869798e-01,  -1.909216869817931e-01,
        -7.395723310447907e-01,  -8.806389019359968e-01,  -3.480296903057334e-02,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1: Test 3 by 5 square matrix (rank = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 6_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_2: Test 3 by 5 square matrix (rank = 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 6_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(3))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 1, 0, 0,
        0, 0, 2, 6, 0,
        0, 0, 3, 8, 7,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_3: Test 3 by 5 square matrix (rank = 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 6_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(3))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         2.702804494912633e-01,   4.335834765383948e-01,  -7.037978928014703e-01,   5.404971423218834e-01,  -8.957298207349242e-01,
        -6.336459944660888e-01,  -1.572273780830846e-02,  -4.308994106802427e-01,   4.312684414455946e-01,   2.107321088246122e-01,
        -1.766070660655237e-01,  -2.685460060684117e-01,   3.904911252586380e-01,   9.197463599879411e-01,   7.483016393778588e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_4: Test 3 by 5 square matrix (rank = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 6_4
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(2))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         3.575095283229116e-01,  0,   1.349169854076371e-02,   3.079265080543283e-01,  -7.355712172801248e-01,
        -8.403328239792179e-02,  0,  -1.533483794820438e-01,  -1.802623733840146e-01,  -3.784613037969309e-02,
         3.575095283229116e-01,  0,   1.349169854076371e-02,   3.079265080543283e-01,  -7.355712172801248e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_1: Test 5 by 5 square matrix (rank = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 7_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_2: Test 5 by 5 square matrix (rank = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 7_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(1))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 8, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_3: Test 5 by 5 square matrix (rank = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 7_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(2))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0, 0, 2, 1, 0,
         0, 0, 0, 0, 0,
         0, 0, 8, 0, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 0, 0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_4: Test 5 by 5 square matrix (rank = 5)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 7_4
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(5))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         2.312019805665849e+05,  -1.631435770629452e+05,   4.580864284699289e+05,   3.417297302926257e+05,  -9.904774296769637e+04,
         8.010156539662811e+04,   2.011079446048353e+05,  -7.733958980904896e+05,   3.009901446248725e+05,   2.985670439695420e+05,
        -5.291431039290484e+04,   6.935223617249326e+05,  -2.776583934642750e+05,   5.097119676992504e+05,   4.288408332881280e+05,
        -2.240529704989486e+05,   3.348556811147994e+05,  -1.890904983167808e+05,   7.488594060067062e+05,  -1.529238757683842e+05,
        -5.117085905285879e+04,   1.582541197017058e+05,   8.029682324063770e+05,   2.128461482805363e+05,   3.565460922674444e+05,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_5: Test 5 by 5 square matrix which contains very small values.
     *
     *       Enter this matrix into the octave rank function and the
     *       rank of the matrix will be equal to 5.
     *
     *       But in this library, very small values ​​will be treated as 0,
     *       so the rank of this matrix will be equal to 0
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 7_5
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(0))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         4.224489546515281e-31,  -2.776389925059539e-31,   6.597375489901119e-31,   6.070780513338758e-33,  -3.538978111860850e-31,
        -7.352108794452528e-31,  -7.805053783268541e-31,  -4.300345665021705e-31,  -9.312860750193786e-32,   3.989596045411336e-31,
        -3.251469428728167e-31,   6.255362186733835e-31,  -4.047449647810635e-31,  -4.289496700669279e-31,  -2.856405041315769e-31,
         2.814671212698542e-31,   5.241770714146821e-32,   3.085900181731433e-31,  -3.943221951606068e-32,   3.544128167720226e-31,
         1.502277847760081e-32,  -2.781031947078702e-31,   4.123993084375791e-31,  -5.514205501433434e-31,  -1.173175237662605e-31,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));

    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "Wrong rank");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
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
LM_UT_CASE_FUNC(lm_ut_lu_invert)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_lu1_decomp = {0};
    lm_mat_t mat_lu1_inverse = {0};
    lm_mat_t mat_a1_mul_inverse1 = {0};
    lm_mat_t mat_inverse1_mul_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_permute_list_t perm_list1 = {0};
    int32_t inv_sgn_det1 = {0};
    lm_mat_elem_size_t rank1 = {0};
    lm_mat_elem_t det1 = {0};

    /*
     * a1_1: Test 2 by 3 non square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 1_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 3,  5,
        7, 9, 11,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*  A must be a square matrix */
    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2: Test 3 by 1 non square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 1_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1,
        3,
        7,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*  A must be a square matrix */
    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_1: Test 1 by 1 square matrix (non-invertible matrix, value = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 2_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_INVERTIBLE)), "");

    /* Validate: the non-invertible matrix should not has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect not full rank");

    /* Validate: the determinant of non-invertible matrix is zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect determinant == 0");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2: Test 2 by 2 square matrix (non-invertible matrix, rank = 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 2_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 1,
        2, 2,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_INVERTIBLE)), "");

    /* Validate: the non-invertible matrix should not has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect not full rank");

    /* Validate: the determinant of non-invertible matrix is zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect determinant == 0");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_3: Test 5 by 5 square matrix (non-invertible matrix, rank = 4)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 2_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0   ,6.687624622164154e-01   ,4.799386131084010e-01   ,4.451484893303013e-01   ,9.644974384252100e-01,
        0.0   ,3.713283996941490e-01   ,1.334860283613037e-01   ,3.269863814573526e-01   ,5.199613292789280e-01,
        0.0   ,9.318435674383434e-02   ,1.393522086844260e-01   ,6.523060444366441e-01   ,7.801718129777306e-01,
        0.0   ,6.668943936435018e-01   ,7.539119969264666e-01   ,5.691800848094646e-02   ,1.458680861319463e-01,
        0.0   ,5.562972939679235e-01   ,1.853191409406154e-01   ,9.899242091486894e-01   ,5.379041124495941e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_INVERTIBLE)), "");

    /* Validate: the non-invertible matrix should not has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect not full rank");

    /* Validate: the determinant of non-invertible matrix is zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect determinant == 0");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1: Test 1 by 1 square matrix (value = -10)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 3_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -10,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2: Test 1 by 1 square matrix (value = 0.5)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 3_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.5,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3: Test 1 by 1 square matrix (value = -123.456789)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 3_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.5,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1: Test 2 by 2 square matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 4_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 2,
        3, 4,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2: Test 2 by 2 square matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 4_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        9.787084215800372e-01,   5.324436989445323e-01,
        8.300018343367362e-01,   7.062642694662653e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_3: Test 2 by 2 square matrix (case 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 4_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.078849064511300e+00,  -9.218149702157302e-02,
        1.045538115202153e+00,   1.255799498900004e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1: Test 3 by 3 square matrix (case 1, lower triangular)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        2,  0,   0,
        8, -7,   0,
        4,  9, -27,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2: Test 3 by 3 square matrix (case 2, upper triangular)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_2
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         2,  4,  6,
         0, -1, -8,
         0,  0, 96,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_3: Test 3 by 3 square matrix (case 3, identity matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_3
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         1,  0,  0,
         0,  1,  0,
         0,  0,  1,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_4: Test 3 by 3 square matrix (case 4)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 5_4
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                3
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        7.744184081847555e-01,   3.933450522334958e-03,  -4.703562700638481e-01,
        3.540277535464731e-02,  -3.332775705208132e-01,   5.850149405731131e-02,
        4.530818859408015e-01,  -1.969815738212495e-01,   4.750897233254969e-02,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1: Test 5 by 5 square matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 6_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        7.698838963084942e-01,   5.808594260982051e-01,   9.045345517450396e-01,   9.378567644598761e-02,   9.530940337616316e-02,
        6.418394735288155e-01,   5.884794978195140e-01,   7.450282276059257e-01,   2.679654397595752e-01,   9.904003788834720e-01,
        9.649946333191102e-01,   1.657430931501970e-01,   9.870638726807535e-01,   5.962903939347046e-01,   2.752841877578115e-01,
        1.329495356200545e-01,   6.249838063607644e-01,   1.952446162914216e-01,   2.626893502672443e-01,   7.328217248655994e-01,
        6.029609917547712e-01,   5.057972113677136e-01,   1.462185731854685e-01,   9.164683892273255e-02,   6.625222699058022e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_1: Test 10 by 10 square matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_DECOMP_NAME
    #undef ELEM_LU_DECOMP_R
    #undef ELEM_LU_DECOMP_C
    #undef ELEM_PERM_LIST
    #undef ELEM_PERM_LIST_SIZE
    #undef RANK_EXPECT

    #define TEST_VAR(var)           var ## 7_1
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                10
    #define ELEM_A_C                10
    #define ELEM_LU_DECOMP_NAME     TEST_VAR(mat_elem_lu_decomp)
    #define ELEM_LU_DECOMP_R        ELEM_A_R
    #define ELEM_LU_DECOMP_C        ELEM_A_C
    #define ELEM_LU_INVERSE_NAME    TEST_VAR(mat_elem_lu_inverse)
    #define ELEM_LU_INVERSE_R       ELEM_A_R
    #define ELEM_LU_INVERSE_C       ELEM_A_C
    #define ELEM_A_MUL_INVERSE_NAME TEST_VAR(mat_elem_a_mul_inverse)
    #define ELEM_A_MUL_INVERSE_R    ELEM_A_R
    #define ELEM_A_MUL_INVERSE_C    ELEM_A_C
    #define ELEM_INVERSE_MUL_A_NAME TEST_VAR(mat_elem_inverse_mul_a)
    #define ELEM_INVERSE_MUL_A_R    ELEM_A_R
    #define ELEM_INVERSE_MUL_A_C    ELEM_A_C
    #define ELEM_PERM_LIST          TEST_VAR(perm_elem_list)
    #define ELEM_PERM_LIST_SIZE     (ELEM_A_R * 2)
    #define RANK_EXPECT             ((lm_mat_elem_size_t)(ELEM_A_R))

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -2.407716494874149e-01, -6.886409935610338e-01,  1.837979763077933e-01,  2.403202771970223e-01, -6.924511877407434e-01,  2.096498440569680e-01, -1.427736571159722e-01, -6.237539725585323e-01, -4.975786110285301e-01, -2.769510176560722e-01,
        -1.499098570627191e-01, -3.497237711487107e-01, -6.452526554919942e-01, -5.870853502818629e-02, -5.587985195775969e-01, -3.130891280812224e-01, -4.198003596830330e-01, -4.201184688050486e-01, -7.297974200560642e-02, -2.315235555706749e-01,
        -5.469118295806459e-01, -6.331552512903095e-01,  1.059192327717119e-01, -2.257638235928596e-01, -4.703686529018985e-01, -5.793942631054425e-01, -4.301739669743603e-01, -2.729081389788442e-01,  5.507519662409144e-03, -2.520343265101934e-01,
        -6.552761832116184e-01, -7.189621004832225e-03, -2.184783038747427e-01,  1.014684041739721e-01, -2.178844086192773e-01, -6.998679299938904e-01, -4.029933017123942e-01, -3.987142247345505e-01,  8.514863448590648e-02, -4.333270607126771e-02,
        -7.127863351919101e-01, -2.027432416433299e-01,  6.881298558805204e-02,  1.347603629226948e-01, -4.973050483003136e-01, -1.553361750772806e-01,  2.056842815869383e-01,  2.264812296617905e-01, -4.107765085688997e-03, -4.621387974739905e-01,
        -1.738604239028698e-01, -6.724115082449287e-01, -3.838708248937295e-01, -4.650753467354706e-01, -2.489754142231357e-01, -7.162200804643222e-01, -6.063734628046726e-01, -2.247496990081658e-01, -2.860942363700067e-01,  2.370148229724854e-01,
        -2.243017567370824e-01, -1.194130180938348e-01, -2.521560795688482e-01, -3.718192329440880e-01, -4.330517922785702e-01,  1.497505069524996e-01, -6.901555029611947e-01,  2.322057080535599e-01,  1.857822315431745e-03, -1.359098806714996e-01,
        -2.342407039218429e-01, -6.641060623863837e-02, -4.396705102203426e-01, -8.161590975751831e-02, -2.125221671689691e-01, -1.442497502600436e-01, -6.518903584675046e-01, -7.235935322678535e-01, -6.235396705563336e-01,  1.646212584839885e-01,
        -1.747473547434597e-01,  2.607726506328977e-01, -3.837483932527276e-01, -1.037479765606609e-01,  2.335650802537226e-01, -2.803830950331529e-01, -2.501942316457714e-01,  1.276976283532484e-01, -1.438763245212302e-01, -1.511098280920253e-02,
        -6.108411140727976e-02, -6.923704351831025e-01, -6.706627475316035e-01, -6.212468184788433e-02, -3.309353723340755e-01, -5.976186646562947e-01, -1.743895441771225e-01, -3.744433957575877e-02, -1.443685058239642e-01, -1.621342616623874e-01,
    };
    lm_mat_elem_t ELEM_LU_DECOMP_NAME[ELEM_LU_DECOMP_R * ELEM_LU_DECOMP_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_INVERSE_NAME[ELEM_LU_INVERSE_R * ELEM_LU_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_A_MUL_INVERSE_NAME[ELEM_A_MUL_INVERSE_R * ELEM_A_MUL_INVERSE_C] = {
        0,
    };
    lm_mat_elem_t ELEM_INVERSE_MUL_A_NAME[ELEM_INVERSE_MUL_A_R * ELEM_INVERSE_MUL_A_C] = {
        0,
    };

    lm_permute_elem_t ELEM_PERM_LIST[ELEM_PERM_LIST_SIZE] = {0};

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C,
                        ELEM_A_NAME, sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_decomp, ELEM_LU_DECOMP_R, ELEM_LU_DECOMP_C,
                        ELEM_LU_DECOMP_NAME, sizeof(ELEM_LU_DECOMP_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1_inverse, ELEM_LU_INVERSE_R, ELEM_LU_INVERSE_C,
                        ELEM_LU_INVERSE_NAME, sizeof(ELEM_LU_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_a1_mul_inverse1, ELEM_A_MUL_INVERSE_R, ELEM_A_MUL_INVERSE_C,
                        ELEM_A_MUL_INVERSE_NAME, sizeof(ELEM_A_MUL_INVERSE_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_inverse1_mul_a1, ELEM_INVERSE_MUL_A_R, ELEM_INVERSE_MUL_A_C,
                        ELEM_INVERSE_MUL_A_NAME, sizeof(ELEM_INVERSE_MUL_A_NAME) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_set(&perm_list1, 0, ELEM_PERM_LIST, ELEM_PERM_LIST_SIZE);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_a1, &mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_decomp(&mat_lu1_decomp, &perm_list1, &inv_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_rank(&mat_lu1_decomp, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_det(&mat_lu1_decomp, inv_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_lu_invert(&mat_lu1_decomp, &perm_list1, &mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate A * inv(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a1, &mat_lu1_inverse,
                          LM_MAT_ZERO_VAL, &mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate inv(A) * A */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_lu1_inverse, &mat_a1,
                          LM_MAT_ZERO_VAL, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: the invertible matrix should has full rank */
    result = lm_chk_elem_almost_equal(rank1, RANK_EXPECT);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect full rank");

    /* Validate: the determinant of invertible matrix is non-zero */
    result = lm_chk_elem_almost_equal(det1, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result != LM_SUCCESS), "expect determinant != 0");

    /* Validate: A * inv(A) == I */
    result = lm_chk_identity_mat(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect A * inv(A) == I");

    /* Validate: inv(A) * A == I */
    result = lm_chk_identity_mat(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "expect inv(A) * A == I");

    /* Validate: inv(A) * A == A * inv(A) */
    result = lm_chk_mat_almost_equal(&mat_a1_mul_inverse1, &mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_SUCCESS), "inv(A) * A == A * inv(A)");

    /* Release resources */
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_decomp);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1_inverse);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_mul_inverse1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_inverse1_mul_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_lu_cases[] =
{
    {"lm_ut_lu_decomp", lm_ut_lu_decomp, NULL, NULL, 0, 0},
    {"lm_ut_lu_det", lm_ut_lu_det, NULL, NULL, 0, 0},
    {"lm_ut_lu_rank", lm_ut_lu_rank, NULL, NULL, 0, 0},
    {"lm_ut_lu_invert", lm_ut_lu_invert, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_lu_suites[] =
{
    {"lm_ut_lu_suites", lm_ut_lu_cases, sizeof(lm_ut_lu_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_list[] =
{
    {lm_ut_lu_suites, sizeof(lm_ut_lu_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_lu()
{
    lm_ut_run(lm_ut_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

