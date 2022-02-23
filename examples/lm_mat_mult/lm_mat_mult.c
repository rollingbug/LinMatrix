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
 * @file    lm_mat_mult.c
 * @brief   Lin matrix, matrix multiplication example code
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
#define LM_MAT_SIZE_C   3


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
    lm_mat_t mat_b = {0};
    lm_mat_t mat_c = {0};

    lm_mat_elem_t elem_a[LM_MAT_SIZE_R * LM_MAT_SIZE_C] = {
        1.0/3.0,    2.0/3.0,    -2.0/3.0,
       -2.0/3.0,    2.0/3.0,     1.0/3.0,
        2.0/3.0,    1.0/3.0,     2.0/3.0
    };
    lm_mat_elem_t elem_b[LM_MAT_SIZE_R * LM_MAT_SIZE_C] = {
        1.0/3.0,   -2.0/3.0,     2.0/3.0,
        2.0/3.0,    2.0/3.0,     1.0/3.0,
       -2.0/3.0,    1.0/3.0,     2.0/3.0
    };
    lm_mat_elem_t elem_c[LM_MAT_SIZE_R * LM_MAT_SIZE_C] = {
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0
    };

    /* Setup required matrix handles for computation */
    result = lm_mat_set(&mat_a, LM_MAT_SIZE_R, LM_MAT_SIZE_C, (lm_mat_elem_t *)elem_a,
                        (sizeof(elem_a) / sizeof(elem_a[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_b, LM_MAT_SIZE_R, LM_MAT_SIZE_C, (lm_mat_elem_t *)elem_b,
                        (sizeof(elem_b) / sizeof(elem_b[0])));
    LM_RETURN_IF_ERR(result);

    result = lm_mat_set(&mat_c, LM_MAT_SIZE_R, LM_MAT_SIZE_C, (lm_mat_elem_t *)elem_c,
                        (sizeof(elem_c) / sizeof(elem_c[0])));
    LM_RETURN_IF_ERR(result);

    /* Compute C := A * B */
    result = lm_oper_gemm(false, false,
                          LM_MAT_ONE_VAL, &mat_a, &mat_b,
                          LM_MAT_ZERO_VAL, &mat_c);
    LM_RETURN_IF_ERR(result);

    printf("\nMatrix A:\n");
    lm_mat_dump(&mat_a);

    printf("\nMatrix B:\n");
    lm_mat_dump(&mat_b);

    printf("\nMatrix A * B:\n");
    lm_mat_dump(&mat_c);

    result = lm_mat_clr(&mat_a);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_b);
    LM_RETURN_IF_ERR(result);

    result = lm_mat_clr(&mat_c);
    LM_RETURN_IF_ERR(result);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

