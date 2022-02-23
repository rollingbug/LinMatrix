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
 * @file    lm_ut_symm_hess.c
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
#include "lm_symm_hess.h"
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
LM_UT_CASE_FUNC(lm_ut_symm_hess)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_symm_hess1 = {0};
    lm_mat_t mat_symm_work1 = {0}; /* M by 1 */
    lm_mat_t mat_t1 = {0};
    lm_mat_t mat_t_work1 = {0}; /* 1 by M */
    lm_mat_t mat_beta1 = {0};
    lm_mat_t mat_q1 = {0};
    lm_mat_t mat_t1_mul_q1 = {0};
    lm_mat_t mat_sim1 = {0}; /* Q' * T * Q */
    lm_mat_elem_t trace1;
    lm_mat_elem_t trace2;

    /*
     * a1_1:
     *      input 3 by 2 non-square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            2
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a1_2:
     *      input 2 by 3 non-square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            3
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_1:
     *      input 1 by 1 matrix (value = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      input 1 by 1 matrix (value = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        2.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      input 2 by 2 matrix (value = 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            2
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      input 2 by 2 matrix (rank 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            2
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0, 2.0,
        0.0, 0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      input 2 by 2 matrix (rank 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            2
    #define ELEM_A_C            2
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,  0.0,
        0.0, -2.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      input 3 by 3 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,  0.0,  0.0,
        0.0,  0.0,  0.0,
        0.0,  0.0,  0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2:
     *      input 3 by 3 matrix (identity matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,  0.0,  0.0,
        0.0,  1.0,  0.0,
        0.0,  0.0,  1.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_3:
     *      input 3 by 3 matrix (symmetric matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,  1.1, -5.0,
        1.1, -0.3, -1.25,
       -5.0, -1.25, 3.33,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_4:
     *      input 3 by 3 matrix (non-symmetric matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 4_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,  1.1, -5.5,
        1.1, -0.3, -2.25,
       -0.5, -1.25, 3.33,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      input 5 by 5 matrix (symmetric matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 5_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.0465875,  -0.1865468,   0.1382564,  -0.1860443,  -0.0011846,
        -0.1865468,  -0.5890798,   0.1876936,  -0.5123769,  -0.2331348,
         0.1382564,   0.1876936,   0.4376957,   0.0131097,   0.5328031,
        -0.1860443,  -0.5123769,   0.0131097,  -0.4003479,  -0.3411601,
        -0.0011846,  -0.2331348,   0.5328031,  -0.3411601,   0.3303318,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1:
     *      input 10 by 10 matrix (symmetric matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 6_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            10
    #define ELEM_A_C            10
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

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
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_2:
     *      input 10 by 10 matrix (non-symmetric matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_SYMM_HESS_NAME
    #undef ELEM_SYMM_HESS_R
    #undef ELEM_SYMM_HESS_C
    #undef ELEM_SYMM_WORK_NAME
    #undef ELEM_SYMM_WORK_R
    #undef ELEM_SYMM_WORK_C
    #undef ELEM_T_NAME
    #undef ELEM_T_R
    #undef ELEM_T_C
    #undef ELEM_T_WORK_NAME
    #undef ELEM_T_WORK_R
    #undef ELEM_T_WORK_C
    #undef ELEM_BETA_NAME
    #undef ELEM_BETA_R
    #undef ELEM_BETA_C
    #undef ELEM_Q_NAME
    #undef ELEM_Q_R
    #undef ELEM_Q_C
    #undef ELEM_T_MUL_Q_NAME
    #undef ELEM_T_MUL_Q_R
    #undef ELEM_T_MUL_Q_C
    #undef ELEM_SIM_NAME
    #undef ELEM_SIM_R
    #undef ELEM_SIM_C

    #define TEST_VAR(var)       var ## 6_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            10
    #define ELEM_A_C            10
    #define ELEM_SYMM_HESS_NAME TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R    ELEM_A_R
    #define ELEM_SYMM_HESS_C    ELEM_A_C
    #define ELEM_SYMM_WORK_NAME TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R    ELEM_A_R
    #define ELEM_SYMM_WORK_C    1
    #define ELEM_T_NAME         TEST_VAR(mat_elem_t)
    #define ELEM_T_R            ELEM_A_R
    #define ELEM_T_C            ELEM_A_C
    #define ELEM_T_WORK_NAME    TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R       1
    #define ELEM_T_WORK_C       ELEM_A_R
    #define ELEM_BETA_NAME      TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R         ELEM_A_R
    #define ELEM_BETA_C         1
    #define ELEM_Q_NAME         TEST_VAR(mat_elem_q)
    #define ELEM_Q_R            ELEM_A_R
    #define ELEM_Q_C            ELEM_A_C
    #define ELEM_T_MUL_Q_NAME   TEST_VAR(mat_elem_t_mul_q)
    #define ELEM_T_MUL_Q_R      ELEM_A_R
    #define ELEM_T_MUL_Q_C      ELEM_A_C
    #define ELEM_SIM_NAME       TEST_VAR(mat_elem_sim)
    #define ELEM_SIM_R          ELEM_A_R
    #define ELEM_SIM_C          ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0.5558224,  -0.1710319,   0.6909785,  -0.6073750,   0.2531603,   0.0254241,  -0.0081630,   0.0755842,  -0.3479231,  -0.0529789,
         0.0825274,   0.1987705,  -0.5051349,   0.0169847,  -0.0483507,  -0.1582026,  -0.0241345,  -0.0167172,  -0.2967136,  -0.2328645,
         0.1599064,   0.0061190,  -0.7965678,  -0.4909346,   0.5533603,  -0.2398303,   0.1806855,  -0.0328520,   0.0543915,   0.0789521,
         0.7956906,   0.5761856,   0.2981841,  -0.0830560,   0.2977616,   0.3314589,  -0.1800306,  -0.4714964,   0.1623617,  -0.3423014,
        -0.6897278,   0.0072041,  -0.4906609,  -0.1822017,  -0.2685408,  -0.0840083,   0.0513331,  -0.0272025,  -0.1226353,   0.1834685,
        -0.6890533,   0.2271838,  -0.3707156,   0.4990950,  -0.1731908,  -0.1792767,   0.1595305,  -0.0644012,  -0.3013819,  -0.2114295,
        -0.7313401,  -0.1029292,  -0.0411772,   0.6638396,   0.3798955,   0.6763401,   0.3519819,   0.2978712,  -0.1195953,   0.3829294,
        -0.2935989,   0.3898868,   0.2006638,   0.3736560,   0.1526575,   0.3706722,   0.3446313,  -0.4868495,   0.2883652,  -0.5554159,
        -0.1834263,  -0.0031934,  -0.8372324,  -0.0518569,   0.0751213,   0.2629230,   0.1320438,  -0.5791252,  -0.1919678,  -0.5072100,
        -0.7205376,  -0.3294575,  -0.4946084,  -0.1521850,  -0.1805984,  -0.0470986,   0.4640013,  -0.1162158,   0.3282675,  -0.4391651,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_NAME[ELEM_SYMM_HESS_R * ELEM_SYMM_HESS_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_WORK_NAME[ELEM_SYMM_WORK_R * ELEM_SYMM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_NAME[ELEM_T_R * ELEM_T_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_WORK_NAME[ELEM_T_WORK_R * ELEM_T_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_BETA_NAME[ELEM_BETA_R * ELEM_BETA_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_Q_NAME[ELEM_Q_R * ELEM_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_T_MUL_Q_NAME[ELEM_T_MUL_Q_R * ELEM_T_MUL_Q_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SIM_NAME[ELEM_SIM_R * ELEM_SIM_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1, ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C, ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C, ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1, ELEM_T_WORK_R, ELEM_T_WORK_C, ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sim1, ELEM_T_MUL_Q_R, ELEM_T_MUL_Q_C, ELEM_T_MUL_Q_NAME,
                        (sizeof(ELEM_T_MUL_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1_mul_q1, ELEM_SIM_R, ELEM_SIM_C, ELEM_SIM_NAME,
                        (sizeof(ELEM_SIM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Q must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The bandwidth of T matrix should be within specified range (tridiagonal matrix) */
    result = lm_chk_banded_mat(&mat_symm_hess1, 1, 1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* The trace of original matrix should equal to the trace of its similar matrix */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_trace(&mat_symm_hess1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check if Q * T * Q' == A */
    result = lm_oper_gemm(false, true,
                          LM_MAT_ONE_VAL, &mat_symm_hess1, &mat_q1,
                          LM_MAT_ZERO_VAL, &mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q1, &mat_t1_mul_q1,
                          LM_MAT_ZERO_VAL, &mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_sim1, &mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_hess1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_beta1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_t1_mul_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sim1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_symm_hess_cases[] =
{
    {"lm_ut_symm_hess", lm_ut_symm_hess, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_symm_hess_suites[] =
{
    {"lm_ut_symm_hess_suites", lm_ut_symm_hess_cases, sizeof(lm_ut_symm_hess_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_symm_hess_list[] =
{
    {lm_ut_symm_hess_suites, sizeof(lm_ut_symm_hess_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_symm_hess()
{
    lm_ut_run(lm_symm_hess_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

