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
 * @file    lm_ut_symm_eigen.c
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
#include "lm_symm_hess.h"
#include "lm_symm_eigen.h"


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
LM_UT_CASE_FUNC(lm_ut_symm_eigen)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_lu1 = {0};
    lm_mat_t mat_symm_hess1 = {0};
    lm_mat_t mat_symm_work1 = {0}; /* M by 1 */
    lm_mat_t mat_t1 = {0};
    lm_mat_t mat_t_work1 = {0}; /* 1 by M */
    lm_mat_t mat_beta1 = {0};
    lm_mat_t mat_q1 = {0};
    lm_mat_t mat_hess_d1 = {0};
    lm_mat_t mat_hess_sd1 = {0};
    lm_mat_t mat_eigen1 = {0};
    lm_mat_t mat_eigen_work1 = {0};
    lm_permute_list_t perm_list1 = {0};
    lm_mat_t mat_diag_shaped = {0};
    lm_mat_t mat_t_subm_shaped = {0};

    lm_mat_elem_t det1;
    lm_mat_elem_t det2;
    lm_mat_elem_size_t rank1;
    lm_mat_elem_size_t rank2;
    lm_mat_elem_t trace1;
    lm_mat_elem_t trace2;
    int32_t invert_sgn_det1;

    /*
     * a2_1:
     *      test 1 by 1 matrix (value = 0)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 2_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, NULL, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 1 by 1 matrix (value = -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 2_2

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.0,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, NULL, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 2 by 2 matrix (zero matrix)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 3_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0,
        0, 0,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 2 by 2 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 3_2

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, -2,
       -2,  3,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      test 2 by 2 matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 3_3

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        8.120376696963909e-01,  7.677980831906360e-01,
        7.677980831906360e-01,  7.259686570595002e-01,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_4:
     *      test 2 by 2 matrix (case 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 3_4

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.459126613543105e+00,  -9.746834758608349e-01,
        -9.746834758608349e-01,  -3.205101957101294e-01,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_5:
     *      test 2 by 2 matrix (case 4, eigenvalue = 1 & -1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 3_5

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0,  1,
        1,  0,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      test 5 by 5 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 5_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                5
    #define ELEM_A_C                5

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -3,    1,   -4,   19,    8,
         1,   -4,   -5,    5,  -11,
        -4,   -5,   15,  -11,   20,
        19,    5,  -11,    6,    3,
         8,  -11,   20,    3,    1,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_clr(&perm_list1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a10_1:
     *      test 10 by 10 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_EIGEN_NAME
    #undef ELEM_EIGEN_R
    #undef ELEM_EIGEN_C
    #undef ELEM_EIGEN_WORK_NAME
    #undef ELEM_EIGEN_WORK_R
    #undef ELEM_EIGEN_WORK_C

    #define TEST_VAR(var)           var ## 10_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                10
    #define ELEM_A_C                10

    /* LU matrix of A, N by N */
    #define ELEM_LU_NAME            TEST_VAR(mat_elem_lu)
    #define ELEM_LU_R               ELEM_A_R
    #define ELEM_LU_C               ELEM_A_C
    #define ELEM_LU_PERM_NAME       TEST_VAR(mat_elem_lu_perm)
    #define ELEM_LU_PERM_R          (ELEM_A_R * 2)
    #define ELEM_LU_PERM_C          1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate V * L * V' (N by N) */
    #define ELEM_EIGEN_NAME         TEST_VAR(mat_elem_hess_eigen)
    #define ELEM_EIGEN_R            ELEM_A_R
    #define ELEM_EIGEN_C            ELEM_A_C
    #define ELEM_EIGEN_WORK_NAME    TEST_VAR(mat_elem_hess_eigen_work)
    #define ELEM_EIGEN_WORK_R       ELEM_A_R
    #define ELEM_EIGEN_WORK_C       ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         4.806864175610754e+02,  2.857238874708495e+02,  8.685701677748521e+01,  3.331897491025284e+02, -1.180065593755924e+02, -8.333333320079376e+01,  6.324101111959703e+02,  4.599561587890226e+02, -1.700493792948176e+02,  8.865814121945072e+02,
         2.857238874708495e+02,  5.199072733750039e+02,  9.202080500117219e+01,  3.164449966583986e+02, -8.088200700820332e+01, -1.285053457084245e+02,  4.463150039511322e+02,  6.764412440929856e+02, -1.809183571144875e+02,  9.441968694885361e+02,
         8.685701677748521e+01,  9.202080500117219e+01,  9.618627966115733e+01,  7.086129606734877e+01, -7.317841556030523e+01, -7.711932255123936e+01,  1.498729736153354e+02,  1.594814217895786e+02, -1.678333212262345e+02,  3.418503401360546e+02,
         3.331897491025284e+02,  3.164449966583986e+02,  7.086129606734877e+01,  2.857238874708495e+02, -8.333333320079375e+01, -8.088200700820332e+01,  4.599561587890226e+02,  4.463150039511322e+02, -1.313116797948236e+02,  7.052848324514990e+02,
        -1.180065593755924e+02, -8.088200700820332e+01, -7.317841556030523e+01, -8.333333320079375e+01,  8.685701677748521e+01,  7.086129606734877e+01, -1.700493792948176e+02, -1.313116797948236e+02,  1.498729736153354e+02, -2.755406116150164e+02,
        -8.333333320079376e+01, -1.285053457084245e+02, -7.711932255123936e+01, -8.088200700820332e+01,  7.086129606734877e+01,  9.202080500117219e+01, -1.313116797948236e+02, -1.809183571144875e+02,  1.594814217895786e+02, -2.944258944318469e+02,
         6.324101111959703e+02,  4.463150039511322e+02,  1.498729736153354e+02,  4.599561587890226e+02, -1.700493792948176e+02, -1.313116797948236e+02,  8.865814121945072e+02,  7.052848324514990e+02, -2.755406116150164e+02,  1.397722222222222e+03,
         4.599561587890226e+02,  6.764412440929856e+02,  1.594814217895786e+02,  4.463150039511322e+02, -1.313116797948236e+02, -1.809183571144875e+02,  7.052848324514990e+02,  9.441968694885361e+02, -2.944258944318468e+02,  1.493761904761905e+03,
        -1.700493792948176e+02, -1.809183571144875e+02, -1.678333212262345e+02, -1.313116797948236e+02,  1.498729736153354e+02,  1.594814217895786e+02, -2.755406116150164e+02, -2.944258944318468e+02,  3.418503401360546e+02, -6.296031746031745e+02,
         8.865814121945072e+02,  9.441968694885361e+02,  3.418503401360546e+02,  7.052848324514990e+02, -2.755406116150164e+02, -2.944258944318469e+02,  1.397722222222222e+03,  1.493761904761905e+03, -6.296031746031745e+02,  3.053000000000000e+03,
    };
    lm_mat_elem_t ELEM_LU_NAME[ELEM_LU_R * ELEM_LU_C] = {
        0.0,
    };
    lm_permute_elem_t ELEM_LU_PERM_NAME[ELEM_LU_PERM_R * ELEM_LU_PERM_C] = {
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_NAME[ELEM_EIGEN_R * ELEM_EIGEN_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_EIGEN_WORK_NAME[ELEM_EIGEN_WORK_R * ELEM_EIGEN_WORK_C] = {
        0.0,
    };

    memcpy((void *)ELEM_LU_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));
    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_lu1, ELEM_LU_R, ELEM_LU_C, ELEM_LU_NAME,
                        (sizeof(ELEM_LU_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen1, ELEM_EIGEN_R, ELEM_EIGEN_C,
                        ELEM_EIGEN_NAME,
                        (sizeof(ELEM_EIGEN_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_eigen_work1, ELEM_EIGEN_WORK_R, ELEM_EIGEN_WORK_C,
                        ELEM_EIGEN_WORK_NAME,
                        (sizeof(ELEM_EIGEN_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_permute_set(&perm_list1,
                            0,
                            ELEM_LU_PERM_NAME,
                            (sizeof(ELEM_LU_PERM_NAME) / sizeof(lm_permute_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of matrix A */
    result = lm_oper_trace(&mat_a1, &trace1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* LU decompose of matrix A, then calculate its rank and determinant */
    result = lm_lu_decomp(&mat_lu1, &perm_list1, &invert_sgn_det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_lu_det(&mat_lu1, invert_sgn_det1, &det1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_lu1, &rank1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_eigen1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the trace of diagonal similarity matrix */
    result = lm_oper_trace(&mat_eigen1, &trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the rank of diagonal similarity matrix */
    result = lm_lu_rank(&mat_eigen1, &rank2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the determinant of diagonal similarity matrix */
    result = lm_lu_det(&mat_eigen1, 1, &det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the Q * L * Q' */
    result = lm_symm_eigen_similar_tf(&mat_q1, &mat_eigen1, &mat_eigen_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate trace 1 == trace 2 */
    result = lm_chk_elem_almost_equal(trace1, trace2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate rank 1 == rank 2 */
    LM_UT_ASSERT((rank1 == rank2), "");

    /* Validate determinant 1 == determinant 2 */
    result = lm_chk_elem_almost_equal(det1, det2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == Q * L * Q' */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_lu1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_eigen_work1);
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
LM_UT_CASE_FUNC(lm_ut_symm_sqrtm)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_symm_hess1 = {0};
    lm_mat_t mat_symm_work1 = {0}; /* M by 1 */
    lm_mat_t mat_t1 = {0};
    lm_mat_t mat_t_work1 = {0}; /* 1 by M */
    lm_mat_t mat_beta1 = {0};
    lm_mat_t mat_q1 = {0};
    lm_mat_t mat_hess_d1 = {0};
    lm_mat_t mat_hess_sd1 = {0};
    lm_mat_t mat_sqrtm1 = {0};
    lm_mat_t mat_sqrtm_work1 = {0};
    lm_mat_t mat_sqrtm_square1 = {0};
    lm_mat_t mat_diag_shaped = {0};
    lm_mat_t mat_t_subm_shaped = {0};

    /*
     * a2_1:
     *      test 1 by 1 matrix (case 1, value = zero)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 2_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, NULL, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 1 by 1 matrix (case 1, value = 9)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 2_2

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         9,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, NULL, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_3:
     *      test 1 by 1 matrix (case 1, value = -100)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 2_3

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                1
    #define ELEM_A_C                1

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         -100,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, NULL, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NEGATIVE)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    //result = lm_oper_gemm(false, false,
    //                      LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
    //                      LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    //result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 2 by 2 matrix (case 1, all zero)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 3_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         0, 0,
         0, 0,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 2 by 2 matrix (case 2, all one)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 3_2

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         1, 1,
         1, 1,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      test 2 by 2 matrix (case 3)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 3_3

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        3.782156430137130e-01, 4.540082068655785e-01,
        4.540082068655785e-01, 5.449892295803971e-01,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_4:
     *      test 2 by 2 matrix
     *      (case 4, one of the eigenvalue is equal to -1,
             the sqrtm function should returns an error)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 3_4

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                2
    #define ELEM_A_C                2

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 1,
        1, 0,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NEGATIVE)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    //result = lm_oper_gemm(false, false,
    //                      LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
    //                      LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    //result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
    //LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      test 3 by 3 matrix
     *      (rank = 3, all eigenvalues are positive)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 4_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                3
    #define ELEM_A_C                3

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         4.115908590405662e-01,   1.227761131190567e-02,  -2.316271943051794e-02,
         1.227761131190567e-02,   4.688217897359092e-01,  -3.606145415673256e-03,
        -2.316271943051794e-02,  -3.606145415673256e-03,   3.946348615671801e-01,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      test 10 by 10 matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C
    #undef ELEM_LU_NAME
    #undef ELEM_LU_R
    #undef ELEM_LU_C
    #undef ELEM_LU_PERM_NAME
    #undef ELEM_LU_PERM_R
    #undef ELEM_LU_PERM_C
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
    #undef ELEM_SYMM_HESS_D_NAME
    #undef ELEM_SYMM_HESS_D_R
    #undef ELEM_SYMM_HESS_D_C
    #undef ELEM_SYMM_HESS_SD_NAME
    #undef ELEM_SYMM_HESS_SD_R
    #undef ELEM_SYMM_HESS_SD_C
    #undef ELEM_SQRTM_NAME
    #undef ELEM_SQRTM_R
    #undef ELEM_SQRTM_C
    #undef ELEM_SQRTM_WORK_NAME
    #undef ELEM_SQRTM_WORK_R
    #undef ELEM_SQRTM_WORK_C
    #undef ELEM_SQRTM_SQUARE_NAME
    #undef ELEM_SQRTM_SQUARE_R
    #undef ELEM_SQRTM_SQUARE_C

    #define TEST_VAR(var)           var ## 5_1

    /* Original matrix A N by N */
    #define ELEM_A_NAME             TEST_VAR(mat_elem_a)
    #define ELEM_A_R                10
    #define ELEM_A_C                10

    /* Symmetric Hessenberg matrix N by N */
    #define ELEM_SYMM_HESS_NAME     TEST_VAR(mat_elem_symm_hess)
    #define ELEM_SYMM_HESS_R        ELEM_A_R
    #define ELEM_SYMM_HESS_C        ELEM_A_C
    #define ELEM_SYMM_WORK_NAME     TEST_VAR(mat_elem_symm_work)
    #define ELEM_SYMM_WORK_R        ELEM_A_R
    #define ELEM_SYMM_WORK_C        1

    /* Explicit symmetric Hessenberg matrix (N by N), A = Q * T * Q' */
    #define ELEM_T_NAME             TEST_VAR(mat_elem_t)
    #define ELEM_T_R                ELEM_A_R
    #define ELEM_T_C                ELEM_A_C
    #define ELEM_T_WORK_NAME        TEST_VAR(mat_elem_t_work)
    #define ELEM_T_WORK_R           1
    #define ELEM_T_WORK_C           ELEM_A_R
    #define ELEM_BETA_NAME          TEST_VAR(mat_elem_beta)
    #define ELEM_BETA_R             ELEM_A_R
    #define ELEM_BETA_C             1
    #define ELEM_Q_NAME             TEST_VAR(mat_elem_q)
    #define ELEM_Q_R                ELEM_A_R
    #define ELEM_Q_C                ELEM_A_C

    /* Main diagonal and sub-diagonal of symmetric Hessenberg matrix */
    #define ELEM_SYMM_HESS_D_NAME   TEST_VAR(mat_elem_hess_d)
    #define ELEM_SYMM_HESS_D_R      ELEM_A_R
    #define ELEM_SYMM_HESS_D_C      1
    #define ELEM_SYMM_HESS_SD_NAME  TEST_VAR(mat_elem_hess_sd)
    #define ELEM_SYMM_HESS_SD_R     (ELEM_A_R - 1)
    #define ELEM_SYMM_HESS_SD_C     1

    /* To calculate sqrtm(A) (N by N) */
    #define ELEM_SQRTM_NAME         TEST_VAR(mat_elem_sqrtm)
    #define ELEM_SQRTM_R            ELEM_A_R
    #define ELEM_SQRTM_C            ELEM_A_C
    #define ELEM_SQRTM_WORK_NAME    TEST_VAR(mat_elem_sqrtm_work)
    #define ELEM_SQRTM_WORK_R       ELEM_A_R
    #define ELEM_SQRTM_WORK_C       ELEM_A_C

    /* To calculate sqrtm(A) * sqrtm(A) (N by N) */
    #define ELEM_SQRTM_SQUARE_NAME  TEST_VAR(mat_elem_sqrtm_square)
    #define ELEM_SQRTM_SQUARE_R     ELEM_A_R
    #define ELEM_SQRTM_SQUARE_C     ELEM_A_C

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
         4.806864175610754e+02,  2.857238874708495e+02,  8.685701677748521e+01,  3.331897491025284e+02, -1.180065593755924e+02, -8.333333320079376e+01,  6.324101111959703e+02,  4.599561587890226e+02, -1.700493792948176e+02,  8.865814121945072e+02,
         2.857238874708495e+02,  5.199072733750039e+02,  9.202080500117219e+01,  3.164449966583986e+02, -8.088200700820332e+01, -1.285053457084245e+02,  4.463150039511322e+02,  6.764412440929856e+02, -1.809183571144875e+02,  9.441968694885361e+02,
         8.685701677748521e+01,  9.202080500117219e+01,  9.618627966115733e+01,  7.086129606734877e+01, -7.317841556030523e+01, -7.711932255123936e+01,  1.498729736153354e+02,  1.594814217895786e+02, -1.678333212262345e+02,  3.418503401360546e+02,
         3.331897491025284e+02,  3.164449966583986e+02,  7.086129606734877e+01,  2.857238874708495e+02, -8.333333320079375e+01, -8.088200700820332e+01,  4.599561587890226e+02,  4.463150039511322e+02, -1.313116797948236e+02,  7.052848324514990e+02,
        -1.180065593755924e+02, -8.088200700820332e+01, -7.317841556030523e+01, -8.333333320079375e+01,  8.685701677748521e+01,  7.086129606734877e+01, -1.700493792948176e+02, -1.313116797948236e+02,  1.498729736153354e+02, -2.755406116150164e+02,
        -8.333333320079376e+01, -1.285053457084245e+02, -7.711932255123936e+01, -8.088200700820332e+01,  7.086129606734877e+01,  9.202080500117219e+01, -1.313116797948236e+02, -1.809183571144875e+02,  1.594814217895786e+02, -2.944258944318469e+02,
         6.324101111959703e+02,  4.463150039511322e+02,  1.498729736153354e+02,  4.599561587890226e+02, -1.700493792948176e+02, -1.313116797948236e+02,  8.865814121945072e+02,  7.052848324514990e+02, -2.755406116150164e+02,  1.397722222222222e+03,
         4.599561587890226e+02,  6.764412440929856e+02,  1.594814217895786e+02,  4.463150039511322e+02, -1.313116797948236e+02, -1.809183571144875e+02,  7.052848324514990e+02,  9.441968694885361e+02, -2.944258944318468e+02,  1.493761904761905e+03,
        -1.700493792948176e+02, -1.809183571144875e+02, -1.678333212262345e+02, -1.313116797948236e+02,  1.498729736153354e+02,  1.594814217895786e+02, -2.755406116150164e+02, -2.944258944318468e+02,  3.418503401360546e+02, -6.296031746031745e+02,
         8.865814121945072e+02,  9.441968694885361e+02,  3.418503401360546e+02,  7.052848324514990e+02, -2.755406116150164e+02, -2.944258944318469e+02,  1.397722222222222e+03,  1.493761904761905e+03, -6.296031746031745e+02,  3.053000000000000e+03,
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
    lm_mat_elem_t ELEM_SYMM_HESS_D_NAME[ELEM_SYMM_HESS_D_R * ELEM_SYMM_HESS_D_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SYMM_HESS_SD_NAME[ELEM_SYMM_HESS_SD_R * ELEM_SYMM_HESS_SD_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_NAME[ELEM_SQRTM_R * ELEM_SQRTM_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_WORK_NAME[ELEM_SQRTM_WORK_R * ELEM_SQRTM_WORK_C] = {
        0.0,
    };
    lm_mat_elem_t ELEM_SQRTM_SQUARE_NAME[ELEM_SQRTM_SQUARE_R * ELEM_SQRTM_SQUARE_C] = {
        0.0,
    };

    memcpy((void *)ELEM_SYMM_HESS_NAME, (void *)ELEM_A_NAME, sizeof(ELEM_A_NAME));

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_hess1,
                        ELEM_SYMM_HESS_R, ELEM_SYMM_HESS_C,
                        ELEM_SYMM_HESS_NAME,
                        (sizeof(ELEM_SYMM_HESS_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_symm_work1, ELEM_SYMM_WORK_R, ELEM_SYMM_WORK_C,
                        ELEM_SYMM_WORK_NAME,
                        (sizeof(ELEM_SYMM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t1, ELEM_T_R, ELEM_T_C, ELEM_T_NAME,
                        (sizeof(ELEM_T_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_t_work1,
                        ELEM_T_WORK_R, ELEM_T_WORK_C,
                        ELEM_T_WORK_NAME,
                        (sizeof(ELEM_T_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_beta1, ELEM_BETA_R, ELEM_BETA_C, ELEM_BETA_NAME,
                        (sizeof(ELEM_BETA_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_q1, ELEM_Q_R, ELEM_Q_C, ELEM_Q_NAME,
                        (sizeof(ELEM_Q_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_d1, ELEM_SYMM_HESS_D_R, ELEM_SYMM_HESS_D_C,
                        ELEM_SYMM_HESS_D_NAME,
                        (sizeof(ELEM_SYMM_HESS_D_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_hess_sd1, ELEM_SYMM_HESS_SD_R, ELEM_SYMM_HESS_SD_C,
                        ELEM_SYMM_HESS_SD_NAME,
                        (sizeof(ELEM_SYMM_HESS_SD_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm1, ELEM_SQRTM_R, ELEM_SQRTM_C,
                        ELEM_SQRTM_NAME,
                        (sizeof(ELEM_SQRTM_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_work1, ELEM_SQRTM_WORK_R, ELEM_SQRTM_WORK_C,
                        ELEM_SQRTM_WORK_NAME,
                        (sizeof(ELEM_SQRTM_WORK_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_set(&mat_sqrtm_square1, ELEM_SQRTM_SQUARE_R, ELEM_SQRTM_SQUARE_C,
                        ELEM_SQRTM_SQUARE_NAME,
                        (sizeof(ELEM_SQRTM_SQUARE_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find the Hessenberg similarity matrix of matrix A */
    result = lm_symm_hess(&mat_symm_hess1, &mat_beta1, &mat_symm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the Q (orthogonal) and T (Hessenberg tridiagonal) */
    result = lm_symm_hess_explicit(&mat_symm_hess1, &mat_beta1, &mat_q1, &mat_t_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Get the main diagonal and sub-diagonal values from Hessenberg matrix T */
    result = lm_shape_diag(&mat_symm_hess1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_symm_hess1, 1, 0,
                                (ELEM_SYMM_HESS_R - 1), (ELEM_SYMM_HESS_C - 1),
                                &mat_t_subm_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_t_subm_shaped, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_diag_shaped, &mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Find out the eigenvalues and eigenvectors of matrix A */
    result = lm_symm_eigen(&mat_hess_d1, &mat_hess_sd1, &mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate: check if the eigenvalues are in ascending order */
    LM_UT_ASSERT((mat_hess_d1.elem.ptr[0]
                  <= mat_hess_d1.elem.ptr[ELEM_SYMM_HESS_D_R - 1]), "");

    /* Q (eigenvectors) must be a orthogonal matrix */
    result = lm_chk_orthogonal_mat(&mat_q1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Put the eigenvalues on the diagonal of a N by N matrix */
    result = lm_shape_diag(&mat_sqrtm1, 0, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_copy(&mat_hess_d1, &mat_diag_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate the sqrtm(A) */
    result = lm_symm_eigen_sqrtm(&mat_q1, &mat_sqrtm1, &mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Calculate sqrtm(A) * sqrtm(A) */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_sqrtm1, &mat_sqrtm1,
                          LM_MAT_ZERO_VAL, &mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Validate A == sqrtm(A) ^ 2 */
    result = lm_chk_mat_almost_equal(&mat_a1, &mat_sqrtm_square1);
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
    result = lm_mat_clr(&mat_hess_d1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_hess_sd1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_work1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_sqrtm_square1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_symm_eigen_cases[] =
{
    {"lm_ut_symm_eigen", lm_ut_symm_eigen, NULL, NULL, 0, 0},
    {"lm_ut_symm_sqrtm", lm_ut_symm_sqrtm, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_symm_eigen_suites[] =
{
    {"lm_ut_symm_eigen_suites", lm_ut_symm_eigen_cases, sizeof(lm_ut_symm_eigen_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_symm_eigen_list[] =
{
    {lm_ut_symm_eigen_suites, sizeof(lm_ut_symm_eigen_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_symm_eigen()
{
    lm_ut_run(lm_symm_eigen_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

