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
 * @file    lm_ut_framework.h
 * @brief   Lin matrix unit test framework header file
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#ifndef __LM_UT_FRAMEWORK_H__
#define __LM_UT_FRAMEWORK_H__

#include <stdint.h>
#include <stdbool.h>


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

#define LM_LOG_PRINT(...)       \
    do {                        \
        printf(__VA_ARGS__);    \
    } while(0)

#define LM_UT_ASSERT(__cond, __msg) \
    do { \
        LM_LOG_PRINT("."); \
        \
        if ((__cond) == false) { \
            LM_LOG_PRINT("\n@ << Unit test >>"); \
            LM_LOG_PRINT("\n@ Assert cond: %s", #__cond); \
            LM_LOG_PRINT("\n@ Message: %s", __msg); \
            LM_LOG_PRINT("\n@ %s:%d (%s)", __FILE__, __LINE__, __FUNCTION__); \
            *p_failure_cnt += 1; \
            fflush(stdout); \
            abort(); \
        } \
        else { \
            *p_success_cnt += 1; \
        } \
    } while (0);

#define LM_UT_CASE_FUNC(__func_name) \
    void __func_name(uint32_t *p_success_cnt, uint32_t *p_failure_cnt)


/*
 *******************************************************************************
 * Data type definition
 *******************************************************************************
 */

typedef void (*lm_ut_func_ptr)(uint32_t *p_success_cnt, uint32_t *p_failure_cnt);

typedef struct lm_ut_case {
    void *p_ut_name;
    lm_ut_func_ptr p_ut_func;
    lm_ut_func_ptr p_setup_func;
    lm_ut_func_ptr p_teardown_func;
    uint32_t success_cnt;
    uint32_t failure_cnt;
} lm_ut_case_t;

typedef struct lm_ut_suite {
    void *p_suite_name;
    lm_ut_case_t *p_cases;
    uint32_t total_case_cnt;
    uint32_t success_case_cnt;
    uint32_t failure_case_cnt;
} lm_ut_suite_t;

typedef struct lm_ut_suite_list {
    lm_ut_suite_t *p_suites;
    uint32_t total_suite_cnt;
    uint32_t success_suite_cnt;
    uint32_t failure_suite_cnt;
} lm_ut_suite_list_t;


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

int32_t lm_ut_run_demo();
int32_t lm_ut_run(lm_ut_suite_list_t *p_list);

/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */


#endif // __LM_UT_FRAMEWORK_H__
