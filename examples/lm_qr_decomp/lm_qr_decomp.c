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
 * @file    lm_qr_decomp.c
 * @brief   Lin matrix QR decomposition example code
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

#define LM_MAT_SIZE_R   3
#define LM_MAT_SIZE_C   2


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
    lm_mat_t mat_a = {0};
    lm_mat_t mat_q = {0};
    lm_mat_t mat_beta = {0};
    lm_mat_t mat_work = {0};
    lm_mat_t mat_qr = {0};

    lm_mat_elem_t elem_a[LM_MAT_SIZE_R * LM_MAT_SIZE_C] = {
        2, 3,
        2, 4,
        1, 1,
    };
    lm_mat_elem_t elem_q[LM_MAT_SIZE_R * LM_MAT_SIZE_R] = {0};
    lm_mat_elem_t elem_beta[LM_MAT_SIZE_R * 1] = {0};
    lm_mat_elem_t elem_work[LM_MAX(LM_MAT_SIZE_R, LM_MAT_SIZE_R) * 1] = {0};
    lm_mat_elem_t elem_qr[LM_MAT_SIZE_R * LM_MAT_SIZE_C] = {0};

    /* Setup required matrix handles for computation */
    result = lm_mat_set(&mat_a, LM_MAT_SIZE_R, LM_MAT_SIZE_C, (lm_mat_elem_t *)elem_a,
                        (sizeof(elem_a) / sizeof(elem_a[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_q, LM_MAT_SIZE_R, LM_MAT_SIZE_R, (lm_mat_elem_t *)elem_q,
                        (sizeof(elem_q) / sizeof(elem_q[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_beta, LM_MAT_SIZE_R, 1, (lm_mat_elem_t *)elem_beta,
                        (sizeof(elem_beta) / sizeof(elem_beta[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work, 1, LM_MAT_SIZE_C, (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_qr, LM_MAT_SIZE_R, LM_MAT_SIZE_C, (lm_mat_elem_t *)elem_qr,
                        (sizeof(elem_qr) / sizeof(elem_qr[0])));
    LM_RETURN_IF_ERR(result);

    printf("\nMatrix A:\n");
    lm_mat_dump(&mat_a);

    /* Perform QR decomposition */
    result = lm_qr_decomp(&mat_a, &mat_beta, &mat_work);
    LM_RETURN_IF_ERR(result);

    /* Setup 1 by M work matrix */
    result = lm_mat_clr(&mat_work);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_work, 1, LM_MAT_SIZE_R, (lm_mat_elem_t *)elem_work,
                        (sizeof(elem_work) / sizeof(elem_work[0])));
    LM_RETURN_IF_ERR(result);

    /* Convert the Q and R to explicit format */
    result = lm_qr_explicit(&mat_a, &mat_beta, &mat_q, &mat_work);
    LM_RETURN_IF_ERR(result);

    printf("\nMatrix Q:\n");
    lm_mat_dump(&mat_q);

    printf("\nMatrix R:\n");
    lm_mat_dump(&mat_a);

    /* Compute Q * R */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_q, &mat_a,
                          LM_MAT_ZERO_VAL, &mat_qr);
    LM_RETURN_IF_ERR(result);

    printf("\nMatrix Q * R:\n");
    lm_mat_dump(&mat_qr);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

