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
 * @file    lm_ut_chk.c
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
#include "lm_mat.h"
#include "lm_chk.h"
#include "lm_err.h"
#include "lm_shape.h"


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
LM_UT_CASE_FUNC(lm_ut_chk_machine_eps)
{
    lm_rtn_t result;
    lm_mat_elem_t not_a_num = (0.0 / 0.0);
    lm_mat_elem_t infinity = (1.0 / 0.0);
    lm_mat_elem_t test_val;

    LM_UT_ASSERT((isnan(not_a_num) != false), "");
    LM_UT_ASSERT((isinf(infinity) != false), "");

    /*
     * Validate: the eps of 1.0, the calculated eps should be equal to
     * FLT_EPSILON or DBL_EPSILON.
     */
    test_val = 1.0;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((test_val == LM_MAT_MACHINE_EPS), "");

    /*
     * Validate: the eps of -1.0, the calculated eps should be equal to
     * FLT_EPSILON or DBL_EPSILON.
     */
    test_val = -1.0;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((test_val == LM_MAT_MACHINE_EPS), "");

    /*
     * Validate: the eps of 500.0, the calculated eps should be larger than
     * FLT_EPSILON or DBL_EPSILON.
     *
     * We are not able to tell what is the correct value of eps(500.0) because
     * the machine epsilon is platform dependent, but we know the eps(500.0)
     * must be larger than eps(1.0)
     */
    test_val = 500.0;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((test_val > LM_MAT_MACHINE_EPS), "");

    /*
     * Validate: the eps of -500.0, the calculated eps should be larger than
     * FLT_EPSILON or DBL_EPSILON.
     */
    test_val = -500.0;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((test_val > LM_MAT_MACHINE_EPS), "");

    /*
     * Validate: the eps of 0.5, the calculated eps should be less than than
     * FLT_EPSILON or DBL_EPSILON.
     *
     * We are not able to tell what is the correct value of eps(0.5) because
     * the machine epsilon is platform dependent, but we know the eps(0.5)
     * must be less than than eps(1.0)
     */
    test_val = 0.5;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((test_val < LM_MAT_MACHINE_EPS), "");

    /*
     * Validate: the eps of -0.5, the calculated eps should be less than than
     * FLT_EPSILON or DBL_EPSILON.
     */
    test_val = -0.5;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((test_val < LM_MAT_MACHINE_EPS), "");

    /*
     * Validate: the eps of NaN
     */
    test_val = not_a_num;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((isnan(test_val) != false), "");

    /*
     * Validate: the eps of Inf
     */
    test_val = infinity;
    result = lm_chk_machine_eps(&test_val);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((isnan(test_val) != false), "");
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
LM_UT_CASE_FUNC(lm_ut_chk_elem_almost_equal)
{
    lm_rtn_t result;
    lm_mat_elem_t not_a_num = (0.0 / 0.0);
    lm_mat_elem_t infinity = (1.0 / 0.0);

    LM_UT_ASSERT((isnan(not_a_num) != false), "");
    LM_UT_ASSERT((isinf(infinity) != false), "");

    /*
     * Compare the "nan" value
     */
    result = lm_chk_elem_almost_equal(not_a_num, not_a_num);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != nan");

    result = lm_chk_elem_almost_equal(not_a_num, (-not_a_num));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != -nan");

    result = lm_chk_elem_almost_equal(not_a_num, infinity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != inf");

    result = lm_chk_elem_almost_equal(not_a_num, -infinity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != -inf");

    result = lm_chk_elem_almost_equal(not_a_num, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != 0.0");

    result = lm_chk_elem_almost_equal((-not_a_num), LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-nan != 0.0");

    result = lm_chk_elem_almost_equal(not_a_num, (-LM_MAT_ZERO_VAL));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != -0.0");

    result = lm_chk_elem_almost_equal((-not_a_num), (-LM_MAT_ZERO_VAL));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-nan != -0.0");

    result = lm_chk_elem_almost_equal(not_a_num, 10.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != -10.0");

    result = lm_chk_elem_almost_equal((-not_a_num), 10.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-nan != -10.0");

    result = lm_chk_elem_almost_equal(not_a_num, (-10.0));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "nan != -10.0");

    result = lm_chk_elem_almost_equal((-not_a_num), (-10.0));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-nan != -10.0");

    /*
     * Compare the "inf" value
     */
    result = lm_chk_elem_almost_equal(infinity, not_a_num);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != nan");

    result = lm_chk_elem_almost_equal(infinity, not_a_num);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-inf != nan");

    result = lm_chk_elem_almost_equal(infinity, infinity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != inf");

    result = lm_chk_elem_almost_equal(infinity, -infinity);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != -inf");

    result = lm_chk_elem_almost_equal(infinity, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != 0.0");

    result = lm_chk_elem_almost_equal(-infinity, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-inf != 0.0");

    result = lm_chk_elem_almost_equal(infinity, -LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != -0.0");

    result = lm_chk_elem_almost_equal(-infinity, -LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-inf != -0.0");

    result = lm_chk_elem_almost_equal(infinity, 10.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != 10.0");

    result = lm_chk_elem_almost_equal(-infinity, 10.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-inf != 10.0");

    result = lm_chk_elem_almost_equal(infinity, -10.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "inf != -10.0");

    result = lm_chk_elem_almost_equal(-infinity, -10.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "-inf != -10.0");

    /*
     * Compare the value "close to 0.0"
     */
    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, LM_MAT_ZERO_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "0 == 0");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, (-LM_MAT_ZERO_VAL));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "0 == -0");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, LM_MAT_EPSILON_MAX);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "0 ~= 0.000000x");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, (-LM_MAT_EPSILON_MAX));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "0 ~= -0.000000x");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, LM_MAT_EPSILON_MAX * 2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "0 != 0.000000x * 2");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, (-LM_MAT_EPSILON_MAX * 2));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "0 != -0.000000x * 2");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "0 != 1");

    result = lm_chk_elem_almost_equal(LM_MAT_ZERO_VAL, (-LM_MAT_ONE_VAL));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "0 != -1");

    /*
     * Compare the value "close to 1"
     */
    result = lm_chk_elem_almost_equal(LM_MAT_ONE_VAL, LM_MAT_EPSILON_MAX);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "1 != 0.000000x");

    result = lm_chk_elem_almost_equal(LM_MAT_ONE_VAL, (-LM_MAT_EPSILON_MAX));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "1 != -0.000000x");

    result = lm_chk_elem_almost_equal(LM_MAT_ONE_VAL, LM_MAT_ONE_VAL);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                  "1 == 1");

    result = lm_chk_elem_almost_equal(LM_MAT_ONE_VAL, (-LM_MAT_ONE_VAL));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "1 != -1");

    /*
     * Compare the large value
     */
    result = lm_chk_elem_almost_equal(100.0, 100.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "100 == 100");

    result = lm_chk_elem_almost_equal(100.0, -100.0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "100 != -100");

    result = lm_chk_elem_almost_equal(100.0, (lm_mat_elem_t)(100.1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "100 == 100.x");

    result = lm_chk_elem_almost_equal(100.0, (lm_mat_elem_t)(-100.1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "100 != -100.x");

    result = lm_chk_elem_almost_equal(100.0, (lm_mat_elem_t)(100.00000000001));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "100 == 100.0000000000x");

    result = lm_chk_elem_almost_equal(100.0, (lm_mat_elem_t)(-100.00000000001));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "100 != -100.000000x");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(9999999999.0), (lm_mat_elem_t)(9999999998.99885));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "9999999999.0 == 9999999998.99885");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(9999999999.0), (lm_mat_elem_t)(-9999999998.99885));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "9999999999.0 != -9999999998.99885");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(9999999.0), (lm_mat_elem_t)(9999999998.99885));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "9999999.0 != 9999999998.99885");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(9999999.0), (lm_mat_elem_t)(-9999999998.99885));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "9999999.0 != -9999999998.99885");

    /* Check small number */
    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(1.0e-15), (lm_mat_elem_t)(1.0e-15 + 1.0e-16));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(-1.0e-15), (lm_mat_elem_t)(-1.0e-15 - 1.0e-16));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(1.0e-15), (lm_mat_elem_t)(1.0e-15 + 1.0e-17));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(-1.0e-15), (lm_mat_elem_t)(-1.0e-15 - 1.0e-17));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(1.0e-15), (lm_mat_elem_t)(1.0e-15 + 1.0e-18));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(-1.0e-15), (lm_mat_elem_t)(-1.0e-15 - 1.0e-18));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check large number */
    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(1.0e+15), (lm_mat_elem_t)(1.0e+15 + 1.0e+1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(-1.0e+15), (lm_mat_elem_t)(-1.0e+15 - 1.0e+1));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(1.0e+15), (lm_mat_elem_t)(1.0e+15 + 1.0e+2));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(-1.0e+15), (lm_mat_elem_t)(-1.0e+15 - 1.0e+2));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(1.0e+15), (lm_mat_elem_t)(1.0e+15 + 1.0e+3));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_elem_almost_equal((lm_mat_elem_t)(-1.0e+15), (lm_mat_elem_t)(-1.0e+15 - 1.0e+3));
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
LM_UT_CASE_FUNC(lm_ut_chk_mat_almost_equal)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a2 = {0};
    lm_mat_t mat_b1 = {0};
    lm_mat_t mat_b2 = {0};
    lm_mat_t mat_c1 = {0};
    lm_mat_t mat_c2 = {0};
    lm_mat_t mat_b1_shaped = {0};
    lm_mat_t mat_b2_shaped = {0};
    lm_mat_elem_t mat_elem_a1[5 * 5] = {
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -0.000000000001,    0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001,
        0.000000000001,     0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001,
        0.000000000001,     0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001,
        0.000000000001,     0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001,
        0.000000000001,     0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001,
    };
    lm_mat_elem_t mat_elem_b1[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_b2[5 * 5] = {
        100.00000000001, 200.00000000001, 300.00000000001, 400.00000000001, 500.00000000001,
        100.00000000001, 200.00000000001, 300.00000000001, 400.00000000001, 500.00000000001,
        100.00000000001, 200.00000000001, 300.00000000001, 400.00000000001, 500.00000000001,
        100.00000000001, 200.00000000001, 300.00000000001, 400.00000000001, 500.00000000001,
        100.00000000001, 200.00000000001, 300.00000000001, 400.00000000001, 500.00000000001,
    };
    lm_mat_elem_t mat_elem_c1[5 * 5] = {
        0.0 / 0.0,  2.0,    3.0,    4.0,    5.0,
        1.0,        2.0,    3.0,    4.0,    5.0,
        1.0,        2.0,    3.0,    4.0,    5.0,
        1.0,        2.0,    3.0,    4.0,    5.0,
        1.0,        2.0,    3.0,    4.0,    5.0,
    };
    lm_mat_elem_t mat_elem_c2[5 * 5] = {
        1.0,    2.0,    3.0,    4.0,    5.0,
        1.0,    2.0,    3.0,    4.0,    5.0,
        1.0,    2.0,    3.0,    4.0,    5.0,
        1.0,    2.0,    3.0,    4.0,    5.0,
        1.0,    2.0,    3.0,    4.0,    5.0,
    };

    /*
     * Test if the two matrices have different row size
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 3, 3,
                        mat_elem_a1, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "Row size mismatched");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if the two matrices have different column size
     */
    result = lm_mat_set(&mat_a1, 3, 3,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_DIM_MISMATCH)), "Column size mismatched");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 1 by 1 matrices are equal
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 1, 1,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "1 by 1 equal");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 5 by 5 matrices are equal
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "5 by 5 equal");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 3 by 5 matrices are equal
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 3, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "3 by 5 equal");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 5 by 3 matrices are equal
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 3, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "5 by 3 equal");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 1 by 5 matrices are equal
     */
    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_a2, 1, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "1 by 5 equal");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 5 by 5 matrices which contains large numbers are equal
     */
    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b1, sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b2, 5, 5,
                        mat_elem_b2, sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1, &mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "large value 5 by 5 equal");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 5 by 5 matrices which contains totally different contents are equal
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b2, 5, 5,
                        mat_elem_b2, sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_a1, &mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "different value 5 by 5 equal");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Check if two 5 by 5 matrices which contains NAN are equal
     */
    result = lm_mat_set(&mat_c1, 5, 5,
                        mat_elem_c1, sizeof(mat_elem_c1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_c2, 5, 5,
                        mat_elem_c2, sizeof(mat_elem_c2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_c1, &mat_c2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_ELEM_VALUE_NOT_EQUAL)),
                 "Not a number 5 by 5 equal");

    result = lm_mat_clr(&mat_c1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_c2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test the re-shaped matrix (sub-matrix)
     */
    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b1, sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b1, 1, 1, 3, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b2, 5, 5,
                        mat_elem_b2, sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_b2, 1, 1, 3, 3, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1_shaped, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "re-shaped 3 by 3 equal (sub-matrix)");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test the re-shaped matrix (diagonal)
     */
    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b1, sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b1, 0, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b2, 5, 5,
                        mat_elem_b2, sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_diag(&mat_b2, 0, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1_shaped, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "re-shaped 5 by 1 equal (diagonal)");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test the re-shaped matrix (row vector)
     */
    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b1, sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_b1, 1, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b2, 5, 5,
                        mat_elem_b2, sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_b2, 1, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1_shaped, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "re-shaped 1 by 5 equal (row vector)");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test the re-shaped matrix (column vector)
     */
    result = lm_mat_set(&mat_b1, 5, 5,
                        mat_elem_b1, sizeof(mat_elem_b1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_b1, 3, &mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_set(&mat_b2, 5, 5,
                        mat_elem_b2, sizeof(mat_elem_b2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_b2, 3, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_mat_almost_equal(&mat_b1_shaped, &mat_b2_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "re-shaped 5 by 1 equal (column vector)");

    result = lm_mat_clr(&mat_b1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_b2_shaped);
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
LM_UT_CASE_FUNC(lm_ut_chk_square_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };

    /*
     * Test if a 5 by 5 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "5 by 5 is square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a 3 by 5 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "3 by 5 is not square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a 5 by 3 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "5 by 3 is not square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a 3 by 5 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 1, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "1 by 5 is not square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a 5 by 3 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 5, 1,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "5 by 1 is not square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a 1 by 1 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "1 by 1 is square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a reshaped 2 by 2 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 2, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)),
                 "reshaped 2 by 2 is square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a reshaped 1 by 2 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 2, 2, 1, 2, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "reshaped 1 by 2 is not square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a single row vector of 5 by 5 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_row_vect(&mat_a1, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "a single row vector of 5 by 5 matrix is not square matrix");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Test if a single column vector of 5 by 5 matrix is a square matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_col_vect(&mat_a1, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_square_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_SQUARE)),
                 "a single column vector of 5 by 5 matrix is not square matrix");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_chk_triu_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        0.0000000000,   200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        0.0000000000,   0.0000000000,   300.0000000000, 400.0000000000, 500.0000000000,
        0.0000000000,   0.0000000000,   0.0000000000,   400.0000000000, 500.0000000000,
        0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,   500.0000000000,
    };
    lm_mat_elem_t mat_elem_a3[5 * 5] = {
        100.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,
        100.0000000000, 200.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 0.0000000000,   0.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 0.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };

    /*
     * Input a 5 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_UPPER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 3 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_UPPER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 3 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_UPPER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 1 by 1 regular matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 5 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 3 by 5 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 1 by 1 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 5 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_UPPER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 3 by 5 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_UPPER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_UPPER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 1 by 1 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_triu_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

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
LM_UT_CASE_FUNC(lm_ut_chk_tril_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        0.0000000000,   200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        0.0000000000,   0.0000000000,   300.0000000000, 400.0000000000, 500.0000000000,
        0.0000000000,   0.0000000000,   0.0000000000,   400.0000000000, 500.0000000000,
        0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,   500.0000000000,
    };
    lm_mat_elem_t mat_elem_a3[5 * 5] = {
        100.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,
        100.0000000000, 200.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 0.0000000000,   0.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 0.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };

    /*
     * Input a 5 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_LOWER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 3 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_LOWER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 3 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 3,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_LOWER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 1 by 1 regular matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 5 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_LOWER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 3 by 5 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_LOWER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_LOWER_TRIANGULAR)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 1 by 1 upper triangular matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 5 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 3 by 5 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 3, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 1 by 1 lower triangular matrix
     */
    result = lm_mat_set(&mat_a1, 1, 1,
                        mat_elem_a3, sizeof(mat_elem_a3) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_tril_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

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
LM_UT_CASE_FUNC(lm_ut_chk_diagonal_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[5 * 5] = {
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        -100.0000000000,    0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,
        0.0000000000,       200.0000000000, 0.0000000000,   0.0000000000,   0.0000000000,
        0.0000000000,       0.0000000000,   300.0000000000, 0.0000000000,   0.0000000000,
        0.0000000000,       0.0000000000,   0.0000000000,   400.0000000000, 0.0000000000,
        0.0000000000,       0.0000000000,   0.0000000000,   0.0000000000,   500.0000000000,
    };

    /*
     * Input a 5 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_DIAGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 3 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 3, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_DIAGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_DIAGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 1 by 1 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 5 diagonal matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 3 by 5 diagonal matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 3, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 diagonal matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 1 by 1 diagonal matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_diagonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_chk_identity_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};
    lm_mat_elem_t mat_elem_a1[5 * 5] = {
          1.0000000000, 200.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000,   1.0000000000, 300.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000,   1.0000000000, 400.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000,   1.0000000000, 500.0000000000,
        100.0000000000, 200.0000000000, 300.0000000000, 400.0000000000,   1.0000000000,
    };
    lm_mat_elem_t mat_elem_a2[5 * 5] = {
        0.999999999999,   0.000000000000,   0.000000000000,   0.000000000000,   0.000000000000,
        0.000000000000,   0.999999999999,   0.000000000000,   0.000000000000,   0.000000000000,
        0.000000000000,   0.000000000000,   1.000000000005,   0.000000000000,   0.000000000000,
        0.000000000000,   0.000000000000,   0.000000000000,   0.999999999999,   0.000000000000,
        0.000000000000,   0.000000000000,   0.000000000000,   0.000000000000,   0.999999999999,
    };

    /*
     * Input a 5 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_IDENTITY)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 3 by 5 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 3, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_IDENTITY)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_IDENTITY)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 1 by 1 regular matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a1, sizeof(mat_elem_a1) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a 5 by 5 identity matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 3 by 5 identity matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 3, 5, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_IDENTITY)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 5 by 3 identity matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 5, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_IDENTITY)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * Input a reshaped 1 by 1 identity matrix
     */
    result = lm_mat_set(&mat_a1, 5, 5,
                        mat_elem_a2, sizeof(mat_elem_a2) / sizeof(lm_mat_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 0, 0, 1, 1, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_identity_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_chk_orthogonal_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};
    lm_mat_t mat_a1_shaped = {0};

    /*
     * a_1_1: test 3 by 1 non-square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 1_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
        0.0,
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a_1_2: test 5 by 3 non-square matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 1_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_1:
     *      test 1 by 1 matrix, value = 0
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 2_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_2:
     *      test 1 by 1 matrix, value = -1
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 2_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_3:
     *      test 1 by 1 matrix, value = 1
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 2_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a2_4:
     *      test 1 by 1 matrix, value = 10
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 2_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        10.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");


    /*
     * a2_5:
     *      test 1 by 1 matrix, value = 0.1
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 2_5
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            1
    #define ELEM_A_C            1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.1,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      test 3 by 3 matrix, zero matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 3_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      test 3 by 3 matrix, identity matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 3_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      test 3 by 3 matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 3_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.428571428571429,       0.875907128252983,      -0.221615056545935,
        -0.285714285714286,      -0.364081878611179,      -0.886460226183741,
        -0.857142857142857,      -0.316592937922765,       0.406294270334215,

    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_4:
     *      test 3 by 3 matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 3_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            3
    #define ELEM_A_C            3

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.4053324844816668,  -0.5863064202300180,  -0.7013917297922617,
        -0.6092768070102889,  -0.3987380643603599,   0.6854120866088524,
        -0.6815330875777542,   0.7051614975501820,  -0.1956014133627828,

    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      test 5 by 5 matrix, zero matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 4_1
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2:
     *      test 5 by 5 matrix, identity matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 4_2
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_3:
     *      test 5 by 5 orthogonal matrix (case 1)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 4_3
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.4969769888613995,  -0.3045481494397725,  -0.5349373779764355,  -0.2548025265214847,  -0.5560413395935873,
        -0.5377684852113507,  -0.3543471315336009,   0.1559063361878384,  -0.3261729187446015,   0.6742014595812668,
        -0.2387217805382837,   0.6508538426419075,  -0.6018497215989832,   0.1539810858338969,   0.3653326221835864,
        -0.4522896983612398,   0.5754883232559104,   0.5440263667053191,  -0.2656269094514261,  -0.3126097820879778,
        -0.4497386675711531,  -0.1639847564899943,   0.1770520853331506,   0.8569825375913990,  -0.0712574197738755,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_4:
     *      test 5 by 5 orthogonal matrix (case 2)
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 4_4
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.3277450917059441,  -0.5025546444599527,   0.7601987044865026,   0.1844997191395583,   0.1675701868736208,
        -0.1882053629940950,  -0.5107686678158777,  -0.3278578982754775,   0.2791416129212989,  -0.7199189314026525,
        -0.8057507856553295,   0.5593943159411707,   0.0367875411970864,   0.0925725565236486,  -0.1670947913607045,
        -0.3799224442673382,  -0.3549686214558097,  -0.2339233157686259,  -0.8088108315676699,   0.1440872485764095,
        -0.2521636452002425,  -0.2182417796477008,  -0.5084620805801692,   0.4746526878577755,   0.6363608726006065,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_5:
     *      test 5 by 5 non orthogonal matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 4_5
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.8797064573425167,   0.2515201590279995,   0.3887651196034536,   0.5117561217414729,   0.2868252406742457,
        0.0399604975339195,   0.8046391355374160,   0.8260245167505347,   0.0245657269914700,   0.7247197361461191,
        0.7069784196178717,   0.2141877941899982,   0.2619912091535320,   0.9258301987453521,   0.0780349470433430,
        0.1227874028981174,   0.7778361839476239,   0.0102276865954996,   0.5614234882867812,   0.7892545930507764,
        0.5538314818856420,   0.5262478643084150,   0.8361452716390528,   0.6752713981583376,   0.2839989641559230,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_6:
     *      test 3 by 3 sub-matrix of 5 by 5 orthogonal matrix
     */
    #undef TEST_VAR
    #undef ELEM_A_NAME
    #undef ELEM_A_R
    #undef ELEM_A_C

    #define TEST_VAR(var)       var ## 4_6
    #define ELEM_A_NAME         TEST_VAR(mat_elem_a)
    #define ELEM_A_R            5
    #define ELEM_A_C            5

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        -0.3277450917059441,  -0.5025546444599527,   0.7601987044865026,   0.1844997191395583,   0.1675701868736208,
        -0.1882053629940950,  -0.5107686678158777,  -0.3278578982754775,   0.2791416129212989,  -0.7199189314026525,
        -0.8057507856553295,   0.5593943159411707,   0.0367875411970864,   0.0925725565236486,  -0.1670947913607045,
        -0.3799224442673382,  -0.3549686214558097,  -0.2339233157686259,  -0.8088108315676699,   0.1440872485764095,
        -0.2521636452002425,  -0.2182417796477008,  -0.5084620805801692,   0.4746526878577755,   0.6363608726006065,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_shape_submatrix(&mat_a1, 1, 1, 3, 3, &mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_chk_orthogonal_mat(&mat_a1_shaped);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_ORTHOGONAL)), "");

    result = lm_mat_clr(&mat_a1);
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
LM_UT_CASE_FUNC(lm_ut_chk_banded_mat)
{
    lm_rtn_t result;
    lm_mat_t mat_a1 = {0};

    /*
     * a2_1:
     *      check 1 by 1 matrix (within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_1:
     *      check 1 by 3 matrix (zero matrix, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_2:
     *      check 1 by 3 matrix (upper BW = 1, not within the specified bandwidth)
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
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, -1.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_BANDED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a3_3:
     *      check 1 by 3 matrix (upper BW = 2, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_1:
     *      check 3 by 1 matrix (zero matrix, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_2:
     *      check 3 by 1 matrix (upper BW = 1, not within the specified bandwidth)
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
    #define EXPECTED_LOW_BW     0
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_BANDED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a4_3:
     *      check 1 by 3 matrix (upper BW = 2, not within the specified bandwidth)
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
    #define EXPECTED_LOW_BW     1
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_BANDED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_1:
     *      check 2 by 3 matrix (zero matrix, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_2:
     *      check 2 by 3 matrix (upper BW = 1, lower BW = 1, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a5_3:
     *      check 2 by 3 matrix (upper BW = 2, lower BW = 0, not within the specified bandwidth)
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
    #define EXPECTED_UP_BW      1

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 0.0, 0.1,
        0.0, 0.0, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_BANDED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_1:
     *      check 3 by 2 matrix (zero matrix, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_2:
     *      check 3 by 2 matrix (upper BW = 1, lower BW = 1, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a6_3:
     *      check 2 by 3 matrix (upper BW = 1, lower BW = 2, not within the specified bandwidth)
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
    #define EXPECTED_UP_BW      0

    lm_mat_elem_t ELEM_A_NAME[ELEM_A_R * ELEM_A_C] = {
        0.0, 1.0,
        0.1, 0.0,
        0.9, 0.0,
    };

    result = lm_mat_set(&mat_a1, ELEM_A_R, ELEM_A_C, ELEM_A_NAME,
                        (sizeof(ELEM_A_NAME) / sizeof(lm_mat_elem_t)));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /* Check the bandwidth */
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_BANDED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_1:
     *      check 5 by 5 matrix (zero matrix, within the specified bandwidth)
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_2:
     *      check 5 by 5 matrix (upper BW = 4, lower BW = 4, within the specified bandwidth)
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
    #define EXPECTED_LOW_BW     100
    #define EXPECTED_UP_BW      100

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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    /*
     * a7_3:
     *      check 5 by 5 matrix (upper BW = 1, lower BW = 1, not within the specified bandwidth)
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
    #define EXPECTED_LOW_BW     0
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
    result = lm_chk_banded_mat(&mat_a1, EXPECTED_LOW_BW, EXPECTED_UP_BW);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_MAT_IS_NOT_BANDED)), "");

    result = lm_mat_clr(&mat_a1);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
}

static lm_ut_case_t lm_ut_chk_cases[] =
{
    {"lm_ut_chk_machine_eps", lm_ut_chk_machine_eps, NULL, NULL, 0, 0},
    {"lm_ut_chk_elem_almost_equal", lm_ut_chk_elem_almost_equal, NULL, NULL, 0, 0},
    {"lm_ut_chk_mat_almost_equal", lm_ut_chk_mat_almost_equal, NULL, NULL, 0, 0},
    {"lm_ut_chk_square_mat", lm_ut_chk_square_mat, NULL, NULL, 0, 0},
    {"lm_ut_chk_triu_mat", lm_ut_chk_triu_mat, NULL, NULL, 0, 0},
    {"lm_ut_chk_tril_mat", lm_ut_chk_tril_mat, NULL, NULL, 0, 0},
    {"lm_ut_chk_diagonal_mat", lm_ut_chk_diagonal_mat, NULL, NULL, 0, 0},
    {"lm_ut_chk_identity_mat", lm_ut_chk_identity_mat, NULL, NULL, 0, 0},
    {"lm_ut_chk_orthogonal_mat", lm_ut_chk_orthogonal_mat, NULL, NULL, 0, 0},
    {"lm_ut_chk_banded_mat", lm_ut_chk_banded_mat, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_chk_suites[] =
{
    {"lm_ut_chk_suites", lm_ut_chk_cases, sizeof(lm_ut_chk_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_list[] =
{
    {lm_ut_chk_suites, sizeof(lm_ut_chk_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_chk()
{
    lm_ut_run(lm_ut_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

