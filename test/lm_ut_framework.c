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
 * @file    lm_ut_framework.c
 * @brief   Lin matrix unit test framework source file
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lm_ut_framework.h"


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
LM_UT_CASE_FUNC(lm_ut_demo_case1)
{
    LM_UT_ASSERT(2 == 1, "Why not?");
}

static lm_ut_case_t lm_ut_demo_cases[] =
{
    {"lm_ut_demo_case1", lm_ut_demo_case1, NULL, NULL, 0, 0},
    {"lm_ut_demo_case2", lm_ut_demo_case1, NULL, NULL, 0, 0}
};

static lm_ut_suite_t lm_ut_demo_suites[] =
{
    {"lm_ut_demo_suite1", lm_ut_demo_cases, sizeof(lm_ut_demo_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_demo_list[] =
{
    {lm_ut_demo_suites, sizeof(lm_ut_demo_suites) / sizeof(lm_ut_suite_t), 0, 0}
};


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
int32_t lm_ut_run_demo()
{
    lm_ut_run(lm_ut_demo_list);

    return 0;
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
int32_t lm_ut_run(lm_ut_suite_list_t *p_list)
{
    uint32_t suite_cnt;
    uint32_t case_cnt;
    lm_ut_suite_t *p_suite;
    lm_ut_case_t *p_case;

    for (suite_cnt = 0; suite_cnt < p_list->total_suite_cnt; suite_cnt++) {

        p_suite = &(p_list->p_suites[suite_cnt]);

        LM_LOG_PRINT("********** Testing Suite << %s >> **********\n",
                     (char *)(p_suite->p_suite_name));

        for (case_cnt = 0; case_cnt < p_suite->total_case_cnt; case_cnt++) {

            p_case = &(p_suite->p_cases[case_cnt]);

            LM_LOG_PRINT("[%03u] [Case %s] ", case_cnt + 1, (char *)(p_case->p_ut_name));

            if (p_case->p_setup_func != NULL) {
                p_case->p_setup_func(&(p_case->success_cnt), &(p_case->failure_cnt));
            }

            if (p_case->p_ut_func != NULL) {
                p_case->p_ut_func(&(p_case->success_cnt), &(p_case->failure_cnt));
            }

            if (p_case->p_teardown_func != NULL) {
                p_case->p_teardown_func(&(p_case->success_cnt), &(p_case->failure_cnt));
            }

            LM_LOG_PRINT("\n");

            if (p_case->failure_cnt != 0) {
                p_suite->failure_case_cnt++;
            }
            else {
                p_suite->success_case_cnt++;
            }
        }

        LM_LOG_PRINT("\nCases (success/failure): %u / %u\n",
                     p_suite->success_case_cnt, p_suite->failure_case_cnt);

        if (p_suite->failure_case_cnt != 0) {
            p_list->failure_suite_cnt++;
        }
        else {
            p_list->success_suite_cnt++;
        }
    }

    LM_LOG_PRINT("Suites (success/failure): %u / %u\n",
                 p_list->success_suite_cnt, p_list->failure_suite_cnt);

    LM_LOG_PRINT("Failure suite list:\n");
    for (suite_cnt = 0; suite_cnt < p_list->total_suite_cnt; suite_cnt++) {

        p_suite = &(p_list->p_suites[suite_cnt]);

        if (p_suite->failure_case_cnt != 0) {
            LM_LOG_PRINT("\t%s\n", (char *)(p_suite->p_suite_name));
        }

    }

    LM_LOG_PRINT("\n\n");

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

