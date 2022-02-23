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
 * @file    lm_ut_oper_norm.c
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
#include "lm_oper_norm.h"


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
LM_UT_CASE_FUNC(lm_ut_oper_norm_fro)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a1: Test 1 by 1 matrix
     */
    lm_mat_elem_t norm1 = 0;
    lm_mat_elem_t norm_expected1 = 1.8;
    lm_mat_elem_t mat_elem_a1[1 * 1] = {
        -1.8,
    };

    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1,
                        sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_norm_fro(&mat_a1, &norm1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm1, norm_expected1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Test 5 by 5 matrix
     */
    lm_mat_elem_t norm2 = 0;
    lm_mat_elem_t norm_expected2 = 179.6524422322168;
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
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

    result = lm_oper_norm_fro(&mat_a1, &norm2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm2, norm_expected2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Test 3 by 5 matrix
     */
    lm_mat_elem_t norm3 = 0;
    lm_mat_elem_t norm_expected3 = 94.68368391650168;
    lm_mat_elem_t mat_elem_a3[3 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
    };

    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3,
                        sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_norm_fro(&mat_a1, &norm3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm3, norm_expected3);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Test 5 by 3 matrix
     */
    lm_mat_elem_t norm4 = 0;
    lm_mat_elem_t norm_expected4 = 1.355359730846390;
    lm_mat_elem_t mat_elem_a4[5 * 3] = {
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

    result = lm_oper_norm_fro(&mat_a1, &norm4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm4, norm_expected4);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5: Test 1 by 5 matrix
     */
    lm_mat_elem_t norm5 = 0;
    lm_mat_elem_t norm_expected5 = 29.24038303442689;
    lm_mat_elem_t mat_elem_a5[1 * 5] = {
        11,  12,  13,  14,  15,
    };

    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a5,
                        sizeof(mat_elem_a5) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_norm_fro(&mat_a1, &norm5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm5, norm_expected5);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6: Test 5 by 1 matrix
     */
    lm_mat_elem_t norm6 = 0;
    lm_mat_elem_t norm_expected6 = 76.19055059520177;
    lm_mat_elem_t mat_elem_a6[5 * 1] = {
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

    result = lm_oper_norm_fro(&mat_a1, &norm6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm6, norm_expected6);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7: Test reshaped 3 by 3 matrix
     */
    lm_mat_elem_t norm7 = 0;
    lm_mat_elem_t norm_expected7 = 102.0147048223931;
    lm_mat_elem_t mat_elem_a7[5 * 5] = {
        11,  12,  13,  14,  15,
        21,  22,  23,  24,  25,
        31,  32,  33,  34,  35,
        41,  42,  43,  44,  45,
        51,  52,  53,  54,  55,
    };

    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a7,
                        sizeof(mat_elem_a7) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_oper_norm_fro(&mat_a1_shaped, &norm7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal(norm7, norm_expected7);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

