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
 * @file    lm_ut_mat.c
 * @brief   Lin matrix unit test
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lm_ut_mat.h"
#include "lm_ut_framework.h"
#include "lm_mat.h"
#include "lm_err.h"
#include "lm_chk.h"


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
LM_UT_CASE_FUNC(lm_ut_mat_set_and_clr)
{
    #define MAT_A_R_SIZE    3
    #define MAT_A_C_SIZE    5
    lm_rtn_t result;
    lm_mat_t mat_a = {0};
    lm_mat_elem_t mat_elem_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,
        1,  1,  1,  1,  1
    };

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        NULL, sizeof(mat_elem_a) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_NULL_PTR)),
                 "Null pointer is not allowed");

    result = lm_mat_set(&mat_a, 0, MAT_A_C_SIZE,
                        mat_elem_a, sizeof(mat_elem_a) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_IS_ZERO)),
                 "Zero dimension is not allowed");

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        mat_elem_a, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_NEED_MORE_MEM)),
                 "The memory buffer size is less then matrix size");

    result = lm_mat_set(&mat_a, 0xFFFF, 0xFFFF,
                        mat_elem_a, 0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_LIMIT_EXCEEDED)),
                 "Matrix dimension limitation");

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        mat_elem_a, sizeof(mat_elem_a) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

#if LM_MAT_NAME_ENABLED
    LM_UT_ASSERT((mat_a.name[0] != 0), "");
#endif // LM_MAT_NAME_ENABLED

    LM_UT_ASSERT((mat_a.stats == (LM_MAT_STATS_INIT)), "");
    LM_UT_ASSERT((mat_a.elem.ptr == mat_elem_a), "");
    LM_UT_ASSERT((mat_a.elem.dim.r == MAT_A_R_SIZE), "");
    LM_UT_ASSERT((mat_a.elem.dim.c == MAT_A_C_SIZE), "");
    LM_UT_ASSERT((mat_a.elem.nxt_r_osf == MAT_A_C_SIZE), "");
    LM_UT_ASSERT((mat_a.mem.ptr == mat_elem_a), "");
    LM_UT_ASSERT((mat_a.mem.elem_tot == (MAT_A_R_SIZE * MAT_A_C_SIZE)), "");
    LM_UT_ASSERT((mat_a.mem.bytes == sizeof(mat_elem_a)), "");

#if LM_MAT_NAME_ENABLED
    result = lm_mat_set_name(&mat_a, "UT");
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    LM_UT_ASSERT((mat_a.name[0] == 'U'), "");
    LM_UT_ASSERT((mat_a.name[1] == 'T'), "");
    LM_UT_ASSERT((mat_a.name[2] == 0), "");
#endif // LM_MAT_NAME_ENABLED

    result = lm_mat_clr(&mat_a);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(NULL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_NULL_PTR)), "");
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
LM_UT_CASE_FUNC(lm_ut_mat_elem_set_and_clr)
{
    lm_rtn_t result;
    lm_mat_t mat_a = {0};
    lm_mat_elem_t elem_val;
    lm_mat_elem_size_t r_idx;
    lm_mat_elem_size_t c_idx;

    /*
     * a1: Access the element that exceeds the row or column size
     */

    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    2

    lm_mat_elem_t mat_elem_a1[MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        1,  1,
        1,  1,
    };

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Out of row size range */
    result = lm_mat_elem_get(&mat_a, 2, 0, &elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_elem_get(&mat_a, 5, 0, &elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_elem_set(&mat_a, 2, 0, elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_elem_set(&mat_a, 5, 0, elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ROW_IDX_OUT_OF_RANGE)), "");

    /* Out of column size range */
    result = lm_mat_elem_get(&mat_a, 0, 2, &elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_elem_get(&mat_a, 0, 6, &elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_elem_set(&mat_a, 0, 2, elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_elem_set(&mat_a, 0, 6, elem_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_COL_IDX_OUT_OF_RANGE)), "");

    result = lm_mat_clr(&mat_a);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2: Set and get the elements in 3 by 3 matrix
     */

    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_A_R_SIZE    3
    #define MAT_A_C_SIZE    3

    lm_mat_elem_t mat_elem_a2[MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        0,
    };

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Set */
    for (r_idx = 0; r_idx < MAT_A_R_SIZE; r_idx++) {
        for (c_idx = 0; c_idx < MAT_A_C_SIZE; c_idx++) {
            result = lm_mat_elem_set(&mat_a, r_idx, c_idx, (r_idx + c_idx));
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        }
    }

    /* Get */
    for (r_idx = 0; r_idx < MAT_A_R_SIZE; r_idx++) {
        for (c_idx = 0; c_idx < MAT_A_C_SIZE; c_idx++) {
            result = lm_mat_elem_get(&mat_a, r_idx, c_idx, &elem_val);
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

            result = lm_chk_elem_almost_equal(elem_val, (r_idx + c_idx));
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
        }
    }

    result = lm_mat_clr(&mat_a);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3: Set and get the elements in 3 by 10 matrix
     */

    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_A_R_SIZE    3
    #define MAT_A_C_SIZE    10

    lm_mat_elem_t mat_elem_a3[MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        0,
    };

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Set */
    for (r_idx = 0; r_idx < MAT_A_R_SIZE; r_idx++) {
        for (c_idx = 0; c_idx < MAT_A_C_SIZE; c_idx++) {
            result = lm_mat_elem_set(&mat_a, r_idx, c_idx, (r_idx + c_idx));
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        }
    }

    /* Get */
    for (r_idx = 0; r_idx < MAT_A_R_SIZE; r_idx++) {
        for (c_idx = 0; c_idx < MAT_A_C_SIZE; c_idx++) {
            result = lm_mat_elem_get(&mat_a, r_idx, c_idx, &elem_val);
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

            result = lm_chk_elem_almost_equal(elem_val, (r_idx + c_idx));
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
        }
    }

    result = lm_mat_clr(&mat_a);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4: Set and get the elements in 16 by 8 matrix
     */

    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_A_R_SIZE    16
    #define MAT_A_C_SIZE    8

    lm_mat_elem_t mat_elem_a4[MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        0,
    };

    result = lm_mat_set(&mat_a, MAT_A_R_SIZE, MAT_A_C_SIZE,
                        mat_elem_a4, sizeof(mat_elem_a4) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Set */
    for (r_idx = 0; r_idx < MAT_A_R_SIZE; r_idx++) {
        for (c_idx = 0; c_idx < MAT_A_C_SIZE; c_idx++) {
            result = lm_mat_elem_set(&mat_a, r_idx, c_idx, (r_idx + c_idx));
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
        }
    }

    /* Get */
    for (r_idx = 0; r_idx < MAT_A_R_SIZE; r_idx++) {
        for (c_idx = 0; c_idx < MAT_A_C_SIZE; c_idx++) {
            result = lm_mat_elem_get(&mat_a, r_idx, c_idx, &elem_val);
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

            result = lm_chk_elem_almost_equal(elem_val, (r_idx + c_idx));
            LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
        }
    }

    result = lm_mat_clr(&mat_a);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_mat_cases[] =
{
    {"lm_ut_mat_set_and_clr", lm_ut_mat_set_and_clr, NULL, NULL, 0, 0},
    {"lm_ut_mat_elem_set_and_clr", lm_ut_mat_elem_set_and_clr, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_mat_suites[] =
{
    {"lm_ut_mat_suites", lm_ut_mat_cases, sizeof(lm_ut_mat_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_list[] =
{
    {lm_ut_mat_suites, sizeof(lm_ut_mat_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_mat()
{
    lm_ut_run(lm_ut_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

