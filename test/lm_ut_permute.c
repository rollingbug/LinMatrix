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
 * @file    lm_ut_permute.c
 * @brief   Lin matrix unit test
 * @author  Y.H.Kuo
 *******************************************************************************
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "lm_ut_permute.h"
#include "lm_ut_framework.h"
#include "lm_permute.h"
#include "lm_err.h"


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
LM_UT_CASE_FUNC(lm_ut_permute_set_and_clr)
{
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    5
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
    };

    result = lm_permute_set(&perm_list_cycle_a,
                            0,
                            NULL,
                            sizeof(perm_list_elem_a) / sizeof(lm_permute_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_NULL_PTR)),
                 "Null pointer is not allowed");

    result = lm_permute_set(&perm_list_cycle_a,
                            0,
                            perm_list_elem_a,
                            0);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_PM_MEM_ELEM_TOTAL_IS_ZERO)),
                 "Zero dimension is not allowed");

    result = lm_permute_set(&perm_list_cycle_a,
                            0,
                            perm_list_elem_a,
                            sizeof(perm_list_elem_a) / sizeof(lm_permute_elem_t));
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    LM_UT_ASSERT((perm_list_cycle_a.stats == (LM_PERMUTE_STATS_INIT)), "");
    LM_UT_ASSERT((perm_list_cycle_a.elem.cyc_grp_num == 0), "");
    LM_UT_ASSERT((perm_list_cycle_a.elem.ptr  == perm_list_elem_a), "");
    LM_UT_ASSERT((perm_list_cycle_a.elem.num == 0), "");
    LM_UT_ASSERT((perm_list_cycle_a.mem.ptr == perm_list_elem_a), "");
    LM_UT_ASSERT((perm_list_cycle_a.mem.elem_tot == (sizeof(perm_list_elem_a) / sizeof(lm_permute_elem_t))), "");

    result = lm_permute_clr(&perm_list_cycle_a);
    LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

    result = lm_permute_clr(NULL);
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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_wrong_oline_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     6
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    3
    int32_t cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 0, 0, 0},
        {2, 2, 2, 0, 0, 0},
        {3, 3, 3, 0, 0, 0},
        {0, 1, 3, 0, 0, 0},
        {4, 5, 6, 0, 0, 0},
    };

    lm_permute_list_t perm_list_oline_a = {0};
    lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                3,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_NEED_MORE_MEM)
                      || result == LM_ERR_CODE(LM_ERR_PM_CYCLE_GROUP_CANNOT_COMPLETE)
                      || result == LM_ERR_CODE(LM_ERR_PM_CYCLE_GROUP_ELEM_NUM_OUT_OF_RANGE)
                      || result == LM_ERR_CODE(LM_ERR_PM_ELEM_VALUE_OUT_OF_RANGE)), "");
    }
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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_wrong_cycle_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     6
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    3
    int32_t cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_cycle_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0,         0,      0,      0,      0,      0},
        {0xFFFF,    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF},
        {0,         1,      2,      3,      4,      5},
        {0, 1,      5,      0xFFFF, 0,      0},
        {5, 5,      5,      0xFFFF, 0,      0},
        {0, 0,      0xFFFF, 7,      0xFFFF, 0},
    };
    lm_permute_size_t perm_list_cycle_group_a[MAT_SAMPLES][1] = {
        {3},    /* 0, 1, 2  */
        {2},    /* 0, 1 2   */
        {2},    /* 0 1, 2   */
        {1},    /* 0 1 2    */
        {1},    /* 0 2 1    */
        {2},    /* 0 2, 1   */
    };
    lm_permute_size_t perm_list_cycle_elem_num_a[MAT_SAMPLES][1] = {
        {6},    /* 0, 1, 2  */
        {5},    /* 0, 1 2   */
        {5},    /* 0 1, 2   */
        {4},    /* 0 1 2    */
        {4},    /* 0 2 1    */
        {5},    /* 0 2, 1   */
    };

    lm_permute_list_t perm_list_oline_a = {0};
    lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        result = lm_permute_set(&perm_list_cycle_a,
                                perm_list_cycle_elem_num_a[cnt][0],
                                perm_list_elem_cycle_a[cnt],
                                sizeof(perm_list_elem_cycle_a[cnt]) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        perm_list_cycle_a.elem.cyc_grp_num = perm_list_cycle_group_a[cnt][0];

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);

        LM_UT_ASSERT((result == LM_ERR_CODE(LM_ERR_PM_ONE_LINE_NOTATION_NUM_MISMATCHED)
                      || result == LM_ERR_CODE(LM_ERR_PM_CYCLE_GROUP_NUM_MISMATCHED)
                      || result == LM_ERR_CODE(LM_ERR_PM_ELEM_VALUE_OUT_OF_RANGE)), "");
    }

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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_1_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     1
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    1
    int32_t cnt;
    lm_permute_size_t group_cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0},            /* 0, */
    };
    lm_permute_elem_t perm_list_elem_cycle_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 0xFFFF},    /* 0, */
    };
    lm_permute_size_t perm_list_cycle_group_a[MAT_SAMPLES][1] = {
        {1},    /* 0, */
    };

    lm_permute_list_t perm_list_oline_a = {0};
    lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {
        result = lm_permute_set(&perm_list_cycle_a,
                                1,
                                perm_list_elem_oline_a[cnt],
                                sizeof(perm_list_elem_oline_a[cnt]) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        LM_UT_ASSERT((perm_list_cycle_a.elem.cyc_grp_num == perm_list_cycle_group_a[cnt][0]),
                     "The reported cycle group number is wrong");

        group_cnt = 0;

        for (elem_cnt = 0; elem_cnt < perm_list_cycle_a.elem.num; elem_cnt++) {

            LM_UT_ASSERT((perm_list_cycle_a.elem.ptr[elem_cnt]
                          == perm_list_elem_cycle_a[cnt][elem_cnt]),
                         "Incorrect cycle notation");

            if (perm_list_cycle_a.elem.ptr[elem_cnt] == LM_PERMUTE_CYCLE_END_SYM) {
                group_cnt++;

                if (group_cnt == perm_list_cycle_a.elem.cyc_grp_num) {
                    break;
                }
            }
        }

        LM_UT_ASSERT((group_cnt == perm_list_cycle_group_a[cnt][0]),
                     "Detected group number is wrong");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }

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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_2_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     2
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    2
    int32_t cnt;
    lm_permute_size_t group_cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 1}, /* 0, 1 */
        {1, 0}, /* 0 1  */
    };
    lm_permute_elem_t perm_list_elem_cycle_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 0xFFFF, 1,      0xFFFF, },  /* 0, 1 */
        {0, 1,      0xFFFF, 0,      },  /* 0 1  */
    };
    lm_permute_size_t perm_list_cycle_group_a[MAT_SAMPLES][1] = {
        {2},    /* 0, 1 */
        {1},    /* 1 0  */
    };

    lm_permute_list_t perm_list_oline_a = {0};
    lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                2,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        LM_UT_ASSERT((perm_list_cycle_a.elem.cyc_grp_num == perm_list_cycle_group_a[cnt][0]),
                     "The reported cycle group number is wrong");

        group_cnt = 0;

        for (elem_cnt = 0; elem_cnt < perm_list_cycle_a.elem.num; elem_cnt++) {

            LM_UT_ASSERT((perm_list_cycle_a.elem.ptr[elem_cnt]
                          == perm_list_elem_cycle_a[cnt][elem_cnt]),
                         "Incorrect cycle notation");

            if (perm_list_cycle_a.elem.ptr[elem_cnt] == LM_PERMUTE_CYCLE_END_SYM) {
                group_cnt++;

                if (group_cnt == perm_list_cycle_a.elem.cyc_grp_num) {
                    break;
                }
            }
        }

        LM_UT_ASSERT((group_cnt == perm_list_cycle_group_a[cnt][0]),
                     "Detected group number is wrong");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }

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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_3_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     6
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    3
    int32_t cnt;
    lm_permute_size_t group_cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 1, 2, 0, 0, 0}, /* 0, 1, 2  */
        {0, 2, 1, 0, 0, 0}, /* 0, 1 2   */
        {1, 0, 2, 0, 0, 0}, /* 0 1, 2   */
        {1, 2, 0, 0, 0, 0}, /* 0 1 2    */
        {2, 0, 1, 0, 0, 0}, /* 0 2 1    */
        {2, 1, 0, 0, 0, 0}, /* 0 2, 1   */
    };
    lm_permute_elem_t perm_list_elem_cycle_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 0xFFFF, 1,      0xFFFF, 2,      0xFFFF},    /* 0, 1, 2  */
        {0, 0xFFFF, 1,      2,      0xFFFF, 0},         /* 0, 1 2   */
        {0, 1,      0xFFFF, 2,      0xFFFF, 0},         /* 0 1, 2   */
        {0, 1,      2,      0xFFFF, 0,      0},         /* 0 1 2    */
        {0, 2,      1,      0xFFFF, 0,      0},         /* 0 2 1    */
        {0, 2,      0xFFFF, 1,      0xFFFF, 0},         /* 0 2, 1   */
    };
    lm_permute_size_t perm_list_cycle_group_a[MAT_SAMPLES][1] = {
        {3},    /* 0, 1, 2  */
        {2},    /* 0, 1 2   */
        {2},    /* 0 1, 2   */
        {1},    /* 0 1 2    */
        {1},    /* 0 2 1    */
        {2},    /* 0 2, 1   */
    };

    lm_permute_list_t perm_list_oline_a = {0};
    lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                3,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        LM_UT_ASSERT((perm_list_cycle_a.elem.cyc_grp_num == perm_list_cycle_group_a[cnt][0]),
                     "The reported cycle group number is wrong");

        group_cnt = 0;

        for (elem_cnt = 0; elem_cnt < perm_list_cycle_a.elem.num; elem_cnt++) {

            LM_UT_ASSERT((perm_list_cycle_a.elem.ptr[elem_cnt]
                          == perm_list_elem_cycle_a[cnt][elem_cnt]),
                         "Incorrect cycle notation");

            if (perm_list_cycle_a.elem.ptr[elem_cnt] == LM_PERMUTE_CYCLE_END_SYM) {
                group_cnt++;

                if (group_cnt == perm_list_cycle_a.elem.cyc_grp_num) {
                    break;
                }
            }
        }

        LM_UT_ASSERT((group_cnt == perm_list_cycle_group_a[cnt][0]),
                     "Detected group number is wrong");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }

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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_5_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     120
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    5
    int32_t cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};

    static lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 1, 2, 3, 4, 0, 0, 0, 0, 0,},
        {0, 1, 2, 4, 3, 0, 0, 0, 0, 0,},
        {0, 1, 3, 2, 4, 0, 0, 0, 0, 0,},
        {0, 1, 3, 4, 2, 0, 0, 0, 0, 0,},
        {0, 1, 4, 2, 3, 0, 0, 0, 0, 0,},
        {0, 1, 4, 3, 2, 0, 0, 0, 0, 0,},
        {0, 2, 1, 3, 4, 0, 0, 0, 0, 0,},
        {0, 2, 1, 4, 3, 0, 0, 0, 0, 0,},
        {0, 2, 3, 1, 4, 0, 0, 0, 0, 0,},
        {0, 2, 3, 4, 1, 0, 0, 0, 0, 0,},
        {0, 2, 4, 1, 3, 0, 0, 0, 0, 0,},
        {0, 2, 4, 3, 1, 0, 0, 0, 0, 0,},
        {0, 3, 1, 2, 4, 0, 0, 0, 0, 0,},
        {0, 3, 1, 4, 2, 0, 0, 0, 0, 0,},
        {0, 3, 2, 1, 4, 0, 0, 0, 0, 0,},
        {0, 3, 2, 4, 1, 0, 0, 0, 0, 0,},
        {0, 3, 4, 1, 2, 0, 0, 0, 0, 0,},
        {0, 3, 4, 2, 1, 0, 0, 0, 0, 0,},
        {0, 4, 1, 2, 3, 0, 0, 0, 0, 0,},
        {0, 4, 1, 3, 2, 0, 0, 0, 0, 0,},
        {0, 4, 2, 1, 3, 0, 0, 0, 0, 0,},
        {0, 4, 2, 3, 1, 0, 0, 0, 0, 0,},
        {0, 4, 3, 1, 2, 0, 0, 0, 0, 0,},
        {0, 4, 3, 2, 1, 0, 0, 0, 0, 0,},
        {1, 0, 2, 3, 4, 0, 0, 0, 0, 0,},
        {1, 0, 2, 4, 3, 0, 0, 0, 0, 0,},
        {1, 0, 3, 2, 4, 0, 0, 0, 0, 0,},
        {1, 0, 3, 4, 2, 0, 0, 0, 0, 0,},
        {1, 0, 4, 2, 3, 0, 0, 0, 0, 0,},
        {1, 0, 4, 3, 2, 0, 0, 0, 0, 0,},
        {1, 2, 0, 3, 4, 0, 0, 0, 0, 0,},
        {1, 2, 0, 4, 3, 0, 0, 0, 0, 0,},
        {1, 2, 3, 0, 4, 0, 0, 0, 0, 0,},
        {1, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 0, 3, 0, 0, 0, 0, 0,},
        {1, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 2, 4, 0, 0, 0, 0, 0,},
        {1, 3, 0, 4, 2, 0, 0, 0, 0, 0,},
        {1, 3, 2, 0, 4, 0, 0, 0, 0, 0,},
        {1, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 0, 2, 0, 0, 0, 0, 0,},
        {1, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 2, 3, 0, 0, 0, 0, 0,},
        {1, 4, 0, 3, 2, 0, 0, 0, 0, 0,},
        {1, 4, 2, 0, 3, 0, 0, 0, 0, 0,},
        {1, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 0, 2, 0, 0, 0, 0, 0,},
        {1, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 3, 4, 0, 0, 0, 0, 0,},
        {2, 0, 1, 4, 3, 0, 0, 0, 0, 0,},
        {2, 0, 3, 1, 4, 0, 0, 0, 0, 0,},
        {2, 0, 3, 4, 1, 0, 0, 0, 0, 0,},
        {2, 0, 4, 1, 3, 0, 0, 0, 0, 0,},
        {2, 0, 4, 3, 1, 0, 0, 0, 0, 0,},
        {2, 1, 0, 3, 4, 0, 0, 0, 0, 0,},
        {2, 1, 0, 4, 3, 0, 0, 0, 0, 0,},
        {2, 1, 3, 0, 4, 0, 0, 0, 0, 0,},
        {2, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 0, 3, 0, 0, 0, 0, 0,},
        {2, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 1, 4, 0, 0, 0, 0, 0,},
        {2, 3, 0, 4, 1, 0, 0, 0, 0, 0,},
        {2, 3, 1, 0, 4, 0, 0, 0, 0, 0,},
        {2, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 0, 1, 0, 0, 0, 0, 0,},
        {2, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 1, 3, 0, 0, 0, 0, 0,},
        {2, 4, 0, 3, 1, 0, 0, 0, 0, 0,},
        {2, 4, 1, 0, 3, 0, 0, 0, 0, 0,},
        {2, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 0, 1, 0, 0, 0, 0, 0,},
        {2, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 2, 4, 0, 0, 0, 0, 0,},
        {3, 0, 1, 4, 2, 0, 0, 0, 0, 0,},
        {3, 0, 2, 1, 4, 0, 0, 0, 0, 0,},
        {3, 0, 2, 4, 1, 0, 0, 0, 0, 0,},
        {3, 0, 4, 1, 2, 0, 0, 0, 0, 0,},
        {3, 0, 4, 2, 1, 0, 0, 0, 0, 0,},
        {3, 1, 0, 2, 4, 0, 0, 0, 0, 0,},
        {3, 1, 0, 4, 2, 0, 0, 0, 0, 0,},
        {3, 1, 2, 0, 4, 0, 0, 0, 0, 0,},
        {3, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 0, 2, 0, 0, 0, 0, 0,},
        {3, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 1, 4, 0, 0, 0, 0, 0,},
        {3, 2, 0, 4, 1, 0, 0, 0, 0, 0,},
        {3, 2, 1, 0, 4, 0, 0, 0, 0, 0,},
        {3, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 0, 1, 0, 0, 0, 0, 0,},
        {3, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 1, 2, 0, 0, 0, 0, 0,},
        {3, 4, 0, 2, 1, 0, 0, 0, 0, 0,},
        {3, 4, 1, 0, 2, 0, 0, 0, 0, 0,},
        {3, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 0, 1, 0, 0, 0, 0, 0,},
        {3, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 2, 3, 0, 0, 0, 0, 0,},
        {4, 0, 1, 3, 2, 0, 0, 0, 0, 0,},
        {4, 0, 2, 1, 3, 0, 0, 0, 0, 0,},
        {4, 0, 2, 3, 1, 0, 0, 0, 0, 0,},
        {4, 0, 3, 1, 2, 0, 0, 0, 0, 0,},
        {4, 0, 3, 2, 1, 0, 0, 0, 0, 0,},
        {4, 1, 0, 2, 3, 0, 0, 0, 0, 0,},
        {4, 1, 0, 3, 2, 0, 0, 0, 0, 0,},
        {4, 1, 2, 0, 3, 0, 0, 0, 0, 0,},
        {4, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 0, 2, 0, 0, 0, 0, 0,},
        {4, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 1, 3, 0, 0, 0, 0, 0,},
        {4, 2, 0, 3, 1, 0, 0, 0, 0, 0,},
        {4, 2, 1, 0, 3, 0, 0, 0, 0, 0,},
        {4, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 0, 1, 0, 0, 0, 0, 0,},
        {4, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 1, 2, 0, 0, 0, 0, 0,},
        {4, 3, 0, 2, 1, 0, 0, 0, 0, 0,},
        {4, 3, 1, 0, 2, 0, 0, 0, 0, 0,},
        {4, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 0, 1, 0, 0, 0, 0, 0,},
        {4, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
    };

    lm_permute_list_t perm_list_oline_a = {0};
    lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                MAT_A_C_SIZE,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }
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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_6_elem)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     720
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    6
    int32_t cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};

    static lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {
        {0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0,},
        {0, 1, 2, 3, 5, 4, 0, 0, 0, 0, 0, 0,},
        {0, 1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0,},
        {0, 1, 2, 4, 5, 3, 0, 0, 0, 0, 0, 0,},
        {0, 1, 2, 5, 3, 4, 0, 0, 0, 0, 0, 0,},
        {0, 1, 2, 5, 4, 3, 0, 0, 0, 0, 0, 0,},
        {0, 1, 3, 2, 4, 5, 0, 0, 0, 0, 0, 0,},
        {0, 1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0,},
        {0, 1, 3, 4, 2, 5, 0, 0, 0, 0, 0, 0,},
        {0, 1, 3, 4, 5, 2, 0, 0, 0, 0, 0, 0,},
        {0, 1, 3, 5, 2, 4, 0, 0, 0, 0, 0, 0,},
        {0, 1, 3, 5, 4, 2, 0, 0, 0, 0, 0, 0,},
        {0, 1, 4, 2, 3, 5, 0, 0, 0, 0, 0, 0,},
        {0, 1, 4, 2, 5, 3, 0, 0, 0, 0, 0, 0,},
        {0, 1, 4, 3, 2, 5, 0, 0, 0, 0, 0, 0,},
        {0, 1, 4, 3, 5, 2, 0, 0, 0, 0, 0, 0,},
        {0, 1, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0,},
        {0, 1, 4, 5, 3, 2, 0, 0, 0, 0, 0, 0,},
        {0, 1, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {0, 1, 5, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {0, 1, 5, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {0, 1, 5, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {0, 1, 5, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {0, 1, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {0, 2, 1, 3, 4, 5, 0, 0, 0, 0, 0, 0,},
        {0, 2, 1, 3, 5, 4, 0, 0, 0, 0, 0, 0,},
        {0, 2, 1, 4, 3, 5, 0, 0, 0, 0, 0, 0,},
        {0, 2, 1, 4, 5, 3, 0, 0, 0, 0, 0, 0,},
        {0, 2, 1, 5, 3, 4, 0, 0, 0, 0, 0, 0,},
        {0, 2, 1, 5, 4, 3, 0, 0, 0, 0, 0, 0,},
        {0, 2, 3, 1, 4, 5, 0, 0, 0, 0, 0, 0,},
        {0, 2, 3, 1, 5, 4, 0, 0, 0, 0, 0, 0,},
        {0, 2, 3, 4, 1, 5, 0, 0, 0, 0, 0, 0,},
        {0, 2, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0,},
        {0, 2, 3, 5, 1, 4, 0, 0, 0, 0, 0, 0,},
        {0, 2, 3, 5, 4, 1, 0, 0, 0, 0, 0, 0,},
        {0, 2, 4, 1, 3, 5, 0, 0, 0, 0, 0, 0,},
        {0, 2, 4, 1, 5, 3, 0, 0, 0, 0, 0, 0,},
        {0, 2, 4, 3, 1, 5, 0, 0, 0, 0, 0, 0,},
        {0, 2, 4, 3, 5, 1, 0, 0, 0, 0, 0, 0,},
        {0, 2, 4, 5, 1, 3, 0, 0, 0, 0, 0, 0,},
        {0, 2, 4, 5, 3, 1, 0, 0, 0, 0, 0, 0,},
        {0, 2, 5, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {0, 2, 5, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {0, 2, 5, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {0, 2, 5, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {0, 2, 5, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {0, 2, 5, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {0, 3, 1, 2, 4, 5, 0, 0, 0, 0, 0, 0,},
        {0, 3, 1, 2, 5, 4, 0, 0, 0, 0, 0, 0,},
        {0, 3, 1, 4, 2, 5, 0, 0, 0, 0, 0, 0,},
        {0, 3, 1, 4, 5, 2, 0, 0, 0, 0, 0, 0,},
        {0, 3, 1, 5, 2, 4, 0, 0, 0, 0, 0, 0,},
        {0, 3, 1, 5, 4, 2, 0, 0, 0, 0, 0, 0,},
        {0, 3, 2, 1, 4, 5, 0, 0, 0, 0, 0, 0,},
        {0, 3, 2, 1, 5, 4, 0, 0, 0, 0, 0, 0,},
        {0, 3, 2, 4, 1, 5, 0, 0, 0, 0, 0, 0,},
        {0, 3, 2, 4, 5, 1, 0, 0, 0, 0, 0, 0,},
        {0, 3, 2, 5, 1, 4, 0, 0, 0, 0, 0, 0,},
        {0, 3, 2, 5, 4, 1, 0, 0, 0, 0, 0, 0,},
        {0, 3, 4, 1, 2, 5, 0, 0, 0, 0, 0, 0,},
        {0, 3, 4, 1, 5, 2, 0, 0, 0, 0, 0, 0,},
        {0, 3, 4, 2, 1, 5, 0, 0, 0, 0, 0, 0,},
        {0, 3, 4, 2, 5, 1, 0, 0, 0, 0, 0, 0,},
        {0, 3, 4, 5, 1, 2, 0, 0, 0, 0, 0, 0,},
        {0, 3, 4, 5, 2, 1, 0, 0, 0, 0, 0, 0,},
        {0, 3, 5, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {0, 3, 5, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {0, 3, 5, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {0, 3, 5, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {0, 3, 5, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {0, 3, 5, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {0, 4, 1, 2, 3, 5, 0, 0, 0, 0, 0, 0,},
        {0, 4, 1, 2, 5, 3, 0, 0, 0, 0, 0, 0,},
        {0, 4, 1, 3, 2, 5, 0, 0, 0, 0, 0, 0,},
        {0, 4, 1, 3, 5, 2, 0, 0, 0, 0, 0, 0,},
        {0, 4, 1, 5, 2, 3, 0, 0, 0, 0, 0, 0,},
        {0, 4, 1, 5, 3, 2, 0, 0, 0, 0, 0, 0,},
        {0, 4, 2, 1, 3, 5, 0, 0, 0, 0, 0, 0,},
        {0, 4, 2, 1, 5, 3, 0, 0, 0, 0, 0, 0,},
        {0, 4, 2, 3, 1, 5, 0, 0, 0, 0, 0, 0,},
        {0, 4, 2, 3, 5, 1, 0, 0, 0, 0, 0, 0,},
        {0, 4, 2, 5, 1, 3, 0, 0, 0, 0, 0, 0,},
        {0, 4, 2, 5, 3, 1, 0, 0, 0, 0, 0, 0,},
        {0, 4, 3, 1, 2, 5, 0, 0, 0, 0, 0, 0,},
        {0, 4, 3, 1, 5, 2, 0, 0, 0, 0, 0, 0,},
        {0, 4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0,},
        {0, 4, 3, 2, 5, 1, 0, 0, 0, 0, 0, 0,},
        {0, 4, 3, 5, 1, 2, 0, 0, 0, 0, 0, 0,},
        {0, 4, 3, 5, 2, 1, 0, 0, 0, 0, 0, 0,},
        {0, 4, 5, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {0, 4, 5, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {0, 4, 5, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {0, 4, 5, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {0, 4, 5, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {0, 4, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
        {0, 5, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {0, 5, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {0, 5, 1, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {0, 5, 1, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {0, 5, 1, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {0, 5, 1, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {0, 5, 2, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {0, 5, 2, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {0, 5, 2, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {0, 5, 2, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {0, 5, 2, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {0, 5, 2, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {0, 5, 3, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {0, 5, 3, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {0, 5, 3, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {0, 5, 3, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {0, 5, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {0, 5, 3, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {0, 5, 4, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {0, 5, 4, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {0, 5, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {0, 5, 4, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {0, 5, 4, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
        {1, 0, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0,},
        {1, 0, 2, 3, 5, 4, 0, 0, 0, 0, 0, 0,},
        {1, 0, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0,},
        {1, 0, 2, 4, 5, 3, 0, 0, 0, 0, 0, 0,},
        {1, 0, 2, 5, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 0, 2, 5, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 0, 3, 2, 4, 5, 0, 0, 0, 0, 0, 0,},
        {1, 0, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0,},
        {1, 0, 3, 4, 2, 5, 0, 0, 0, 0, 0, 0,},
        {1, 0, 3, 4, 5, 2, 0, 0, 0, 0, 0, 0,},
        {1, 0, 3, 5, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 0, 3, 5, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 0, 4, 2, 3, 5, 0, 0, 0, 0, 0, 0,},
        {1, 0, 4, 2, 5, 3, 0, 0, 0, 0, 0, 0,},
        {1, 0, 4, 3, 2, 5, 0, 0, 0, 0, 0, 0,},
        {1, 0, 4, 3, 5, 2, 0, 0, 0, 0, 0, 0,},
        {1, 0, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 0, 4, 5, 3, 2, 0, 0, 0, 0, 0, 0,},
        {1, 0, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 0, 5, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 0, 5, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 0, 5, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 0, 5, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 0, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {1, 2, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0,},
        {1, 2, 0, 3, 5, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 0, 4, 3, 5, 0, 0, 0, 0, 0, 0,},
        {1, 2, 0, 4, 5, 3, 0, 0, 0, 0, 0, 0,},
        {1, 2, 0, 5, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 0, 5, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 2, 3, 0, 4, 5, 0, 0, 0, 0, 0, 0,},
        {1, 2, 3, 0, 5, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 3, 4, 0, 5, 0, 0, 0, 0, 0, 0,},
        {1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0,},
        {1, 2, 3, 5, 0, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 3, 5, 4, 0, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 0, 3, 5, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 0, 5, 3, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 3, 0, 5, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 3, 5, 0, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 5, 0, 3, 0, 0, 0, 0, 0, 0,},
        {1, 2, 4, 5, 3, 0, 0, 0, 0, 0, 0, 0,},
        {1, 2, 5, 0, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 5, 0, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 2, 5, 3, 0, 4, 0, 0, 0, 0, 0, 0,},
        {1, 2, 5, 3, 4, 0, 0, 0, 0, 0, 0, 0,},
        {1, 2, 5, 4, 0, 3, 0, 0, 0, 0, 0, 0,},
        {1, 2, 5, 4, 3, 0, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 2, 4, 5, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 2, 5, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 4, 2, 5, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 4, 5, 2, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 5, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 0, 5, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 3, 2, 0, 4, 5, 0, 0, 0, 0, 0, 0,},
        {1, 3, 2, 0, 5, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 2, 4, 0, 5, 0, 0, 0, 0, 0, 0,},
        {1, 3, 2, 4, 5, 0, 0, 0, 0, 0, 0, 0,},
        {1, 3, 2, 5, 0, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 0, 2, 5, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 0, 5, 2, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 2, 0, 5, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 2, 5, 0, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 5, 0, 2, 0, 0, 0, 0, 0, 0,},
        {1, 3, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0,},
        {1, 3, 5, 0, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 5, 0, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 3, 5, 2, 0, 4, 0, 0, 0, 0, 0, 0,},
        {1, 3, 5, 2, 4, 0, 0, 0, 0, 0, 0, 0,},
        {1, 3, 5, 4, 0, 2, 0, 0, 0, 0, 0, 0,},
        {1, 3, 5, 4, 2, 0, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 2, 3, 5, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 2, 5, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 3, 2, 5, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 3, 5, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 5, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 0, 5, 3, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 2, 0, 3, 5, 0, 0, 0, 0, 0, 0,},
        {1, 4, 2, 0, 5, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 2, 3, 0, 5, 0, 0, 0, 0, 0, 0,},
        {1, 4, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0,},
        {1, 4, 2, 5, 0, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 2, 5, 3, 0, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 0, 2, 5, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 0, 5, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 2, 0, 5, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 5, 0, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 3, 5, 2, 0, 0, 0, 0, 0, 0, 0,},
        {1, 4, 5, 0, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 5, 0, 3, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 5, 2, 0, 3, 0, 0, 0, 0, 0, 0,},
        {1, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0,},
        {1, 4, 5, 3, 0, 2, 0, 0, 0, 0, 0, 0,},
        {1, 4, 5, 3, 2, 0, 0, 0, 0, 0, 0, 0,},
        {1, 5, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 5, 0, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 5, 0, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 5, 0, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 5, 0, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 5, 0, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {1, 5, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0,},
        {1, 5, 2, 0, 4, 3, 0, 0, 0, 0, 0, 0,},
        {1, 5, 2, 3, 0, 4, 0, 0, 0, 0, 0, 0,},
        {1, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0,},
        {1, 5, 2, 4, 0, 3, 0, 0, 0, 0, 0, 0,},
        {1, 5, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0,},
        {1, 5, 3, 0, 2, 4, 0, 0, 0, 0, 0, 0,},
        {1, 5, 3, 0, 4, 2, 0, 0, 0, 0, 0, 0,},
        {1, 5, 3, 2, 0, 4, 0, 0, 0, 0, 0, 0,},
        {1, 5, 3, 2, 4, 0, 0, 0, 0, 0, 0, 0,},
        {1, 5, 3, 4, 0, 2, 0, 0, 0, 0, 0, 0,},
        {1, 5, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0,},
        {1, 5, 4, 0, 2, 3, 0, 0, 0, 0, 0, 0,},
        {1, 5, 4, 0, 3, 2, 0, 0, 0, 0, 0, 0,},
        {1, 5, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0,},
        {1, 5, 4, 2, 3, 0, 0, 0, 0, 0, 0, 0,},
        {1, 5, 4, 3, 0, 2, 0, 0, 0, 0, 0, 0,},
        {1, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 3, 4, 5, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 3, 5, 4, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 4, 3, 5, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 4, 5, 3, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 5, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 0, 1, 5, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 0, 3, 1, 4, 5, 0, 0, 0, 0, 0, 0,},
        {2, 0, 3, 1, 5, 4, 0, 0, 0, 0, 0, 0,},
        {2, 0, 3, 4, 1, 5, 0, 0, 0, 0, 0, 0,},
        {2, 0, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0,},
        {2, 0, 3, 5, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 0, 3, 5, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 0, 4, 1, 3, 5, 0, 0, 0, 0, 0, 0,},
        {2, 0, 4, 1, 5, 3, 0, 0, 0, 0, 0, 0,},
        {2, 0, 4, 3, 1, 5, 0, 0, 0, 0, 0, 0,},
        {2, 0, 4, 3, 5, 1, 0, 0, 0, 0, 0, 0,},
        {2, 0, 4, 5, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 0, 4, 5, 3, 1, 0, 0, 0, 0, 0, 0,},
        {2, 0, 5, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 0, 5, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 0, 5, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 0, 5, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 0, 5, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 0, 5, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {2, 1, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0,},
        {2, 1, 0, 3, 5, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 0, 4, 3, 5, 0, 0, 0, 0, 0, 0,},
        {2, 1, 0, 4, 5, 3, 0, 0, 0, 0, 0, 0,},
        {2, 1, 0, 5, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 0, 5, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 1, 3, 0, 4, 5, 0, 0, 0, 0, 0, 0,},
        {2, 1, 3, 0, 5, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 3, 4, 0, 5, 0, 0, 0, 0, 0, 0,},
        {2, 1, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0,},
        {2, 1, 3, 5, 0, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 3, 5, 4, 0, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 0, 3, 5, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 0, 5, 3, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 3, 0, 5, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 3, 5, 0, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 5, 0, 3, 0, 0, 0, 0, 0, 0,},
        {2, 1, 4, 5, 3, 0, 0, 0, 0, 0, 0, 0,},
        {2, 1, 5, 0, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 5, 0, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 1, 5, 3, 0, 4, 0, 0, 0, 0, 0, 0,},
        {2, 1, 5, 3, 4, 0, 0, 0, 0, 0, 0, 0,},
        {2, 1, 5, 4, 0, 3, 0, 0, 0, 0, 0, 0,},
        {2, 1, 5, 4, 3, 0, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 1, 4, 5, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 4, 1, 5, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 4, 5, 1, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 5, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 0, 5, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 3, 1, 0, 4, 5, 0, 0, 0, 0, 0, 0,},
        {2, 3, 1, 0, 5, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 1, 4, 0, 5, 0, 0, 0, 0, 0, 0,},
        {2, 3, 1, 4, 5, 0, 0, 0, 0, 0, 0, 0,},
        {2, 3, 1, 5, 0, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 0, 1, 5, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 0, 5, 1, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 1, 0, 5, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 1, 5, 0, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 5, 0, 1, 0, 0, 0, 0, 0, 0,},
        {2, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0, 0,},
        {2, 3, 5, 0, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 5, 0, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 3, 5, 1, 0, 4, 0, 0, 0, 0, 0, 0,},
        {2, 3, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0,},
        {2, 3, 5, 4, 0, 1, 0, 0, 0, 0, 0, 0,},
        {2, 3, 5, 4, 1, 0, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 1, 3, 5, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 1, 5, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 3, 1, 5, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 3, 5, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 5, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 0, 5, 3, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 1, 0, 3, 5, 0, 0, 0, 0, 0, 0,},
        {2, 4, 1, 0, 5, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 1, 3, 0, 5, 0, 0, 0, 0, 0, 0,},
        {2, 4, 1, 3, 5, 0, 0, 0, 0, 0, 0, 0,},
        {2, 4, 1, 5, 0, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 1, 5, 3, 0, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 0, 1, 5, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 0, 5, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 1, 0, 5, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 1, 5, 0, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 5, 0, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 3, 5, 1, 0, 0, 0, 0, 0, 0, 0,},
        {2, 4, 5, 0, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 5, 0, 3, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 5, 1, 0, 3, 0, 0, 0, 0, 0, 0,},
        {2, 4, 5, 1, 3, 0, 0, 0, 0, 0, 0, 0,},
        {2, 4, 5, 3, 0, 1, 0, 0, 0, 0, 0, 0,},
        {2, 4, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0,},
        {2, 5, 0, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 5, 0, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 5, 0, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 5, 0, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 5, 0, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 5, 0, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {2, 5, 1, 0, 3, 4, 0, 0, 0, 0, 0, 0,},
        {2, 5, 1, 0, 4, 3, 0, 0, 0, 0, 0, 0,},
        {2, 5, 1, 3, 0, 4, 0, 0, 0, 0, 0, 0,},
        {2, 5, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0,},
        {2, 5, 1, 4, 0, 3, 0, 0, 0, 0, 0, 0,},
        {2, 5, 1, 4, 3, 0, 0, 0, 0, 0, 0, 0,},
        {2, 5, 3, 0, 1, 4, 0, 0, 0, 0, 0, 0,},
        {2, 5, 3, 0, 4, 1, 0, 0, 0, 0, 0, 0,},
        {2, 5, 3, 1, 0, 4, 0, 0, 0, 0, 0, 0,},
        {2, 5, 3, 1, 4, 0, 0, 0, 0, 0, 0, 0,},
        {2, 5, 3, 4, 0, 1, 0, 0, 0, 0, 0, 0,},
        {2, 5, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0,},
        {2, 5, 4, 0, 1, 3, 0, 0, 0, 0, 0, 0,},
        {2, 5, 4, 0, 3, 1, 0, 0, 0, 0, 0, 0,},
        {2, 5, 4, 1, 0, 3, 0, 0, 0, 0, 0, 0,},
        {2, 5, 4, 1, 3, 0, 0, 0, 0, 0, 0, 0,},
        {2, 5, 4, 3, 0, 1, 0, 0, 0, 0, 0, 0,},
        {2, 5, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 2, 4, 5, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 2, 5, 4, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 4, 2, 5, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 4, 5, 2, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 5, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 0, 1, 5, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 0, 2, 1, 4, 5, 0, 0, 0, 0, 0, 0,},
        {3, 0, 2, 1, 5, 4, 0, 0, 0, 0, 0, 0,},
        {3, 0, 2, 4, 1, 5, 0, 0, 0, 0, 0, 0,},
        {3, 0, 2, 4, 5, 1, 0, 0, 0, 0, 0, 0,},
        {3, 0, 2, 5, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 0, 2, 5, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 0, 4, 1, 2, 5, 0, 0, 0, 0, 0, 0,},
        {3, 0, 4, 1, 5, 2, 0, 0, 0, 0, 0, 0,},
        {3, 0, 4, 2, 1, 5, 0, 0, 0, 0, 0, 0,},
        {3, 0, 4, 2, 5, 1, 0, 0, 0, 0, 0, 0,},
        {3, 0, 4, 5, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 0, 4, 5, 2, 1, 0, 0, 0, 0, 0, 0,},
        {3, 0, 5, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 0, 5, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 0, 5, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 0, 5, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 0, 5, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 0, 5, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {3, 1, 0, 2, 4, 5, 0, 0, 0, 0, 0, 0,},
        {3, 1, 0, 2, 5, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 0, 4, 2, 5, 0, 0, 0, 0, 0, 0,},
        {3, 1, 0, 4, 5, 2, 0, 0, 0, 0, 0, 0,},
        {3, 1, 0, 5, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 0, 5, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 1, 2, 0, 4, 5, 0, 0, 0, 0, 0, 0,},
        {3, 1, 2, 0, 5, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 2, 4, 0, 5, 0, 0, 0, 0, 0, 0,},
        {3, 1, 2, 4, 5, 0, 0, 0, 0, 0, 0, 0,},
        {3, 1, 2, 5, 0, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 0, 2, 5, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 0, 5, 2, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 2, 0, 5, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 2, 5, 0, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 5, 0, 2, 0, 0, 0, 0, 0, 0,},
        {3, 1, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0,},
        {3, 1, 5, 0, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 5, 0, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 1, 5, 2, 0, 4, 0, 0, 0, 0, 0, 0,},
        {3, 1, 5, 2, 4, 0, 0, 0, 0, 0, 0, 0,},
        {3, 1, 5, 4, 0, 2, 0, 0, 0, 0, 0, 0,},
        {3, 1, 5, 4, 2, 0, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 1, 4, 5, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 4, 1, 5, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 4, 5, 1, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 5, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 0, 5, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 2, 1, 0, 4, 5, 0, 0, 0, 0, 0, 0,},
        {3, 2, 1, 0, 5, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 1, 4, 0, 5, 0, 0, 0, 0, 0, 0,},
        {3, 2, 1, 4, 5, 0, 0, 0, 0, 0, 0, 0,},
        {3, 2, 1, 5, 0, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 0, 1, 5, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 0, 5, 1, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 1, 0, 5, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 1, 5, 0, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 5, 0, 1, 0, 0, 0, 0, 0, 0,},
        {3, 2, 4, 5, 1, 0, 0, 0, 0, 0, 0, 0,},
        {3, 2, 5, 0, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 5, 0, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 2, 5, 1, 0, 4, 0, 0, 0, 0, 0, 0,},
        {3, 2, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0,},
        {3, 2, 5, 4, 0, 1, 0, 0, 0, 0, 0, 0,},
        {3, 2, 5, 4, 1, 0, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 1, 2, 5, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 1, 5, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 2, 1, 5, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 2, 5, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 5, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 0, 5, 2, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 1, 0, 2, 5, 0, 0, 0, 0, 0, 0,},
        {3, 4, 1, 0, 5, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 1, 2, 0, 5, 0, 0, 0, 0, 0, 0,},
        {3, 4, 1, 2, 5, 0, 0, 0, 0, 0, 0, 0,},
        {3, 4, 1, 5, 0, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 1, 5, 2, 0, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 0, 1, 5, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 0, 5, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 1, 0, 5, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 5, 0, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0,},
        {3, 4, 5, 0, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 5, 0, 2, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 5, 1, 0, 2, 0, 0, 0, 0, 0, 0,},
        {3, 4, 5, 1, 2, 0, 0, 0, 0, 0, 0, 0,},
        {3, 4, 5, 2, 0, 1, 0, 0, 0, 0, 0, 0,},
        {3, 4, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0,},
        {3, 5, 0, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 5, 0, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 5, 0, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 5, 0, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 5, 0, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 5, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {3, 5, 1, 0, 2, 4, 0, 0, 0, 0, 0, 0,},
        {3, 5, 1, 0, 4, 2, 0, 0, 0, 0, 0, 0,},
        {3, 5, 1, 2, 0, 4, 0, 0, 0, 0, 0, 0,},
        {3, 5, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0,},
        {3, 5, 1, 4, 0, 2, 0, 0, 0, 0, 0, 0,},
        {3, 5, 1, 4, 2, 0, 0, 0, 0, 0, 0, 0,},
        {3, 5, 2, 0, 1, 4, 0, 0, 0, 0, 0, 0,},
        {3, 5, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0,},
        {3, 5, 2, 1, 0, 4, 0, 0, 0, 0, 0, 0,},
        {3, 5, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0,},
        {3, 5, 2, 4, 0, 1, 0, 0, 0, 0, 0, 0,},
        {3, 5, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0,},
        {3, 5, 4, 0, 1, 2, 0, 0, 0, 0, 0, 0,},
        {3, 5, 4, 0, 2, 1, 0, 0, 0, 0, 0, 0,},
        {3, 5, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0,},
        {3, 5, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0,},
        {3, 5, 4, 2, 0, 1, 0, 0, 0, 0, 0, 0,},
        {3, 5, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 2, 3, 5, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 2, 5, 3, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 3, 2, 5, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 3, 5, 2, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 5, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 0, 1, 5, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 0, 2, 1, 3, 5, 0, 0, 0, 0, 0, 0,},
        {4, 0, 2, 1, 5, 3, 0, 0, 0, 0, 0, 0,},
        {4, 0, 2, 3, 1, 5, 0, 0, 0, 0, 0, 0,},
        {4, 0, 2, 3, 5, 1, 0, 0, 0, 0, 0, 0,},
        {4, 0, 2, 5, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 0, 2, 5, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 0, 3, 1, 2, 5, 0, 0, 0, 0, 0, 0,},
        {4, 0, 3, 1, 5, 2, 0, 0, 0, 0, 0, 0,},
        {4, 0, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0,},
        {4, 0, 3, 2, 5, 1, 0, 0, 0, 0, 0, 0,},
        {4, 0, 3, 5, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 0, 3, 5, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 0, 5, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 0, 5, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 0, 5, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 0, 5, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 0, 5, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 0, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 1, 0, 2, 3, 5, 0, 0, 0, 0, 0, 0,},
        {4, 1, 0, 2, 5, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 0, 3, 2, 5, 0, 0, 0, 0, 0, 0,},
        {4, 1, 0, 3, 5, 2, 0, 0, 0, 0, 0, 0,},
        {4, 1, 0, 5, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 0, 5, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 1, 2, 0, 3, 5, 0, 0, 0, 0, 0, 0,},
        {4, 1, 2, 0, 5, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 2, 3, 0, 5, 0, 0, 0, 0, 0, 0,},
        {4, 1, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0,},
        {4, 1, 2, 5, 0, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 2, 5, 3, 0, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 0, 2, 5, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 0, 5, 2, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 2, 0, 5, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 5, 0, 2, 0, 0, 0, 0, 0, 0,},
        {4, 1, 3, 5, 2, 0, 0, 0, 0, 0, 0, 0,},
        {4, 1, 5, 0, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 5, 0, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 1, 5, 2, 0, 3, 0, 0, 0, 0, 0, 0,},
        {4, 1, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0,},
        {4, 1, 5, 3, 0, 2, 0, 0, 0, 0, 0, 0,},
        {4, 1, 5, 3, 2, 0, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 1, 3, 5, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 1, 5, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 3, 1, 5, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 3, 5, 1, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 5, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 0, 5, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 2, 1, 0, 3, 5, 0, 0, 0, 0, 0, 0,},
        {4, 2, 1, 0, 5, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 1, 3, 0, 5, 0, 0, 0, 0, 0, 0,},
        {4, 2, 1, 3, 5, 0, 0, 0, 0, 0, 0, 0,},
        {4, 2, 1, 5, 0, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 1, 5, 3, 0, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 0, 1, 5, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 0, 5, 1, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 1, 0, 5, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 1, 5, 0, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 5, 0, 1, 0, 0, 0, 0, 0, 0,},
        {4, 2, 3, 5, 1, 0, 0, 0, 0, 0, 0, 0,},
        {4, 2, 5, 0, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 5, 0, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 2, 5, 1, 0, 3, 0, 0, 0, 0, 0, 0,},
        {4, 2, 5, 1, 3, 0, 0, 0, 0, 0, 0, 0,},
        {4, 2, 5, 3, 0, 1, 0, 0, 0, 0, 0, 0,},
        {4, 2, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 1, 2, 5, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 1, 5, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 2, 1, 5, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 2, 5, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 5, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 0, 5, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 1, 0, 2, 5, 0, 0, 0, 0, 0, 0,},
        {4, 3, 1, 0, 5, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 1, 2, 0, 5, 0, 0, 0, 0, 0, 0,},
        {4, 3, 1, 2, 5, 0, 0, 0, 0, 0, 0, 0,},
        {4, 3, 1, 5, 0, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 1, 5, 2, 0, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 0, 1, 5, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 0, 5, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 1, 0, 5, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 5, 0, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0,},
        {4, 3, 5, 0, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 5, 0, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 5, 1, 0, 2, 0, 0, 0, 0, 0, 0,},
        {4, 3, 5, 1, 2, 0, 0, 0, 0, 0, 0, 0,},
        {4, 3, 5, 2, 0, 1, 0, 0, 0, 0, 0, 0,},
        {4, 3, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0,},
        {4, 5, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 5, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 5, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 5, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 5, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 5, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 5, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0,},
        {4, 5, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0,},
        {4, 5, 1, 2, 0, 3, 0, 0, 0, 0, 0, 0,},
        {4, 5, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0,},
        {4, 5, 1, 3, 0, 2, 0, 0, 0, 0, 0, 0,},
        {4, 5, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0,},
        {4, 5, 2, 0, 1, 3, 0, 0, 0, 0, 0, 0,},
        {4, 5, 2, 0, 3, 1, 0, 0, 0, 0, 0, 0,},
        {4, 5, 2, 1, 0, 3, 0, 0, 0, 0, 0, 0,},
        {4, 5, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0,},
        {4, 5, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0,},
        {4, 5, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0,},
        {4, 5, 3, 0, 1, 2, 0, 0, 0, 0, 0, 0,},
        {4, 5, 3, 0, 2, 1, 0, 0, 0, 0, 0, 0,},
        {4, 5, 3, 1, 0, 2, 0, 0, 0, 0, 0, 0,},
        {4, 5, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0,},
        {4, 5, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0,},
        {4, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0,},
        {5, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {5, 0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {5, 0, 1, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {5, 0, 1, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {5, 0, 1, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {5, 0, 1, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {5, 0, 2, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {5, 0, 2, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {5, 0, 2, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {5, 0, 2, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {5, 0, 2, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {5, 0, 2, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {5, 0, 3, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {5, 0, 3, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {5, 0, 3, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {5, 0, 3, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {5, 0, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {5, 0, 3, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {5, 0, 4, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {5, 0, 4, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {5, 0, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {5, 0, 4, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {5, 0, 4, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {5, 0, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
        {5, 1, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0,},
        {5, 1, 0, 2, 4, 3, 0, 0, 0, 0, 0, 0,},
        {5, 1, 0, 3, 2, 4, 0, 0, 0, 0, 0, 0,},
        {5, 1, 0, 3, 4, 2, 0, 0, 0, 0, 0, 0,},
        {5, 1, 0, 4, 2, 3, 0, 0, 0, 0, 0, 0,},
        {5, 1, 0, 4, 3, 2, 0, 0, 0, 0, 0, 0,},
        {5, 1, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0,},
        {5, 1, 2, 0, 4, 3, 0, 0, 0, 0, 0, 0,},
        {5, 1, 2, 3, 0, 4, 0, 0, 0, 0, 0, 0,},
        {5, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0,},
        {5, 1, 2, 4, 0, 3, 0, 0, 0, 0, 0, 0,},
        {5, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0,},
        {5, 1, 3, 0, 2, 4, 0, 0, 0, 0, 0, 0,},
        {5, 1, 3, 0, 4, 2, 0, 0, 0, 0, 0, 0,},
        {5, 1, 3, 2, 0, 4, 0, 0, 0, 0, 0, 0,},
        {5, 1, 3, 2, 4, 0, 0, 0, 0, 0, 0, 0,},
        {5, 1, 3, 4, 0, 2, 0, 0, 0, 0, 0, 0,},
        {5, 1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0,},
        {5, 1, 4, 0, 2, 3, 0, 0, 0, 0, 0, 0,},
        {5, 1, 4, 0, 3, 2, 0, 0, 0, 0, 0, 0,},
        {5, 1, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0,},
        {5, 1, 4, 2, 3, 0, 0, 0, 0, 0, 0, 0,},
        {5, 1, 4, 3, 0, 2, 0, 0, 0, 0, 0, 0,},
        {5, 1, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0,},
        {5, 2, 0, 1, 3, 4, 0, 0, 0, 0, 0, 0,},
        {5, 2, 0, 1, 4, 3, 0, 0, 0, 0, 0, 0,},
        {5, 2, 0, 3, 1, 4, 0, 0, 0, 0, 0, 0,},
        {5, 2, 0, 3, 4, 1, 0, 0, 0, 0, 0, 0,},
        {5, 2, 0, 4, 1, 3, 0, 0, 0, 0, 0, 0,},
        {5, 2, 0, 4, 3, 1, 0, 0, 0, 0, 0, 0,},
        {5, 2, 1, 0, 3, 4, 0, 0, 0, 0, 0, 0,},
        {5, 2, 1, 0, 4, 3, 0, 0, 0, 0, 0, 0,},
        {5, 2, 1, 3, 0, 4, 0, 0, 0, 0, 0, 0,},
        {5, 2, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0,},
        {5, 2, 1, 4, 0, 3, 0, 0, 0, 0, 0, 0,},
        {5, 2, 1, 4, 3, 0, 0, 0, 0, 0, 0, 0,},
        {5, 2, 3, 0, 1, 4, 0, 0, 0, 0, 0, 0,},
        {5, 2, 3, 0, 4, 1, 0, 0, 0, 0, 0, 0,},
        {5, 2, 3, 1, 0, 4, 0, 0, 0, 0, 0, 0,},
        {5, 2, 3, 1, 4, 0, 0, 0, 0, 0, 0, 0,},
        {5, 2, 3, 4, 0, 1, 0, 0, 0, 0, 0, 0,},
        {5, 2, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0,},
        {5, 2, 4, 0, 1, 3, 0, 0, 0, 0, 0, 0,},
        {5, 2, 4, 0, 3, 1, 0, 0, 0, 0, 0, 0,},
        {5, 2, 4, 1, 0, 3, 0, 0, 0, 0, 0, 0,},
        {5, 2, 4, 1, 3, 0, 0, 0, 0, 0, 0, 0,},
        {5, 2, 4, 3, 0, 1, 0, 0, 0, 0, 0, 0,},
        {5, 2, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0,},
        {5, 3, 0, 1, 2, 4, 0, 0, 0, 0, 0, 0,},
        {5, 3, 0, 1, 4, 2, 0, 0, 0, 0, 0, 0,},
        {5, 3, 0, 2, 1, 4, 0, 0, 0, 0, 0, 0,},
        {5, 3, 0, 2, 4, 1, 0, 0, 0, 0, 0, 0,},
        {5, 3, 0, 4, 1, 2, 0, 0, 0, 0, 0, 0,},
        {5, 3, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0,},
        {5, 3, 1, 0, 2, 4, 0, 0, 0, 0, 0, 0,},
        {5, 3, 1, 0, 4, 2, 0, 0, 0, 0, 0, 0,},
        {5, 3, 1, 2, 0, 4, 0, 0, 0, 0, 0, 0,},
        {5, 3, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0,},
        {5, 3, 1, 4, 0, 2, 0, 0, 0, 0, 0, 0,},
        {5, 3, 1, 4, 2, 0, 0, 0, 0, 0, 0, 0,},
        {5, 3, 2, 0, 1, 4, 0, 0, 0, 0, 0, 0,},
        {5, 3, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0,},
        {5, 3, 2, 1, 0, 4, 0, 0, 0, 0, 0, 0,},
        {5, 3, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0,},
        {5, 3, 2, 4, 0, 1, 0, 0, 0, 0, 0, 0,},
        {5, 3, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0,},
        {5, 3, 4, 0, 1, 2, 0, 0, 0, 0, 0, 0,},
        {5, 3, 4, 0, 2, 1, 0, 0, 0, 0, 0, 0,},
        {5, 3, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0,},
        {5, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0,},
        {5, 3, 4, 2, 0, 1, 0, 0, 0, 0, 0, 0,},
        {5, 3, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0,},
        {5, 4, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0,},
        {5, 4, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0,},
        {5, 4, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0,},
        {5, 4, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0,},
        {5, 4, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0,},
        {5, 4, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0,},
        {5, 4, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0,},
        {5, 4, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0,},
        {5, 4, 1, 2, 0, 3, 0, 0, 0, 0, 0, 0,},
        {5, 4, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0,},
        {5, 4, 1, 3, 0, 2, 0, 0, 0, 0, 0, 0,},
        {5, 4, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0,},
        {5, 4, 2, 0, 1, 3, 0, 0, 0, 0, 0, 0,},
        {5, 4, 2, 0, 3, 1, 0, 0, 0, 0, 0, 0,},
        {5, 4, 2, 1, 0, 3, 0, 0, 0, 0, 0, 0,},
        {5, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0,},
        {5, 4, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0,},
        {5, 4, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0,},
        {5, 4, 3, 0, 1, 2, 0, 0, 0, 0, 0, 0,},
        {5, 4, 3, 0, 2, 1, 0, 0, 0, 0, 0, 0,},
        {5, 4, 3, 1, 0, 2, 0, 0, 0, 0, 0, 0,},
        {5, 4, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0,},
        {5, 4, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0,},
        {5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0,},
    };

    lm_permute_list_t perm_list_oline_a = {0};
    static lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    static lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                MAT_A_C_SIZE,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }
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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_512_elem_1)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     1
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    512
    int32_t cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {{
         0,     1,     2,     3,     4,     5,     6,     7,     8,     9,
         10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
         20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
         30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
         40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
         50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
         60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
         70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
         80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
         90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
        100,   101,   102,   103,   104,   105,   106,   107,   108,   109,
        110,   111,   112,   113,   114,   115,   116,   117,   118,   119,
        120,   121,   122,   123,   124,   125,   126,   127,   128,   129,
        130,   131,   132,   133,   134,   135,   136,   137,   138,   139,
        140,   141,   142,   143,   144,   145,   146,   147,   148,   149,
        150,   151,   152,   153,   154,   155,   156,   157,   158,   159,
        160,   161,   162,   163,   164,   165,   166,   167,   168,   169,
        170,   171,   172,   173,   174,   175,   176,   177,   178,   179,
        180,   181,   182,   183,   184,   185,   186,   187,   188,   189,
        190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
        200,   201,   202,   203,   204,   205,   206,   207,   208,   209,
        210,   211,   212,   213,   214,   215,   216,   217,   218,   219,
        220,   221,   222,   223,   224,   225,   226,   227,   228,   229,
        230,   231,   232,   233,   234,   235,   236,   237,   238,   239,
        240,   241,   242,   243,   244,   245,   246,   247,   248,   249,
        250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
        260,   261,   262,   263,   264,   265,   266,   267,   268,   269,
        270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
        280,   281,   282,   283,   284,   285,   286,   287,   288,   289,
        290,   291,   292,   293,   294,   295,   296,   297,   298,   299,
        300,   301,   302,   303,   304,   305,   306,   307,   308,   309,
        310,   311,   312,   313,   314,   315,   316,   317,   318,   319,
        320,   321,   322,   323,   324,   325,   326,   327,   328,   329,
        330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
        340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
        350,   351,   352,   353,   354,   355,   356,   357,   358,   359,
        360,   361,   362,   363,   364,   365,   366,   367,   368,   369,
        370,   371,   372,   373,   374,   375,   376,   377,   378,   379,
        380,   381,   382,   383,   384,   385,   386,   387,   388,   389,
        390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
        400,   401,   402,   403,   404,   405,   406,   407,   408,   409,
        410,   411,   412,   413,   414,   415,   416,   417,   418,   419,
        420,   421,   422,   423,   424,   425,   426,   427,   428,   429,
        430,   431,   432,   433,   434,   435,   436,   437,   438,   439,
        440,   441,   442,   443,   444,   445,   446,   447,   448,   449,
        450,   451,   452,   453,   454,   455,   456,   457,   458,   459,
        460,   461,   462,   463,   464,   465,   466,   467,   468,   469,
        470,   471,   472,   473,   474,   475,   476,   477,   478,   479,
        480,   481,   482,   483,   484,   485,   486,   487,   488,   489,
        490,   491,   492,   493,   494,   495,   496,   497,   498,   499,
        500,   501,   502,   503,   504,   505,   506,   507,   508,   509,
        510,   511,

        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,
    }};

    lm_permute_list_t perm_list_oline_a = {0};
    static lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    static lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                MAT_A_C_SIZE,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }
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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_512_elem_2)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     1
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    512
    int32_t cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {{
          0,     1,     3,     9,     4,     5,     6,     7,     8,     2,
         10,    11,    13,    19,    14,    15,    16,    17,    18,    12,
         20,    21,    23,    29,    24,    25,    26,    27,    28,    22,
         30,    31,    33,    39,    34,    35,    36,    37,    38,    32,
         40,    41,    43,    49,    44,    45,    46,    47,    48,    42,
         50,    51,    53,    59,    54,    55,    56,    57,    58,    52,
         60,    61,    63,    69,    64,    65,    66,    67,    68,    62,
         70,    71,    73,    79,    74,    75,    76,    77,    78,    72,
         80,    81,    83,    89,    84,    85,    86,    87,    88,    82,
         90,    91,    93,    99,    94,    95,    96,    97,    98,    92,
        160,   161,   163,   169,   164,   165,   166,   167,   168,   162,
        170,   171,   173,   179,   174,   175,   176,   177,   178,   172,
        180,   181,   183,   189,   184,   185,   186,   187,   188,   182,
        190,   191,   193,   199,   194,   195,   196,   197,   198,   192,
        200,   201,   203,   209,   204,   205,   206,   207,   208,   202,
        210,   211,   213,   219,   214,   215,   216,   217,   218,   212,
        220,   221,   223,   229,   224,   225,   226,   227,   228,   222,
        230,   231,   233,   239,   234,   235,   236,   237,   238,   232,
        240,   241,   243,   249,   244,   245,   246,   247,   248,   242,
        250,   251,   253,   259,   254,   255,   256,   257,   258,   252,
        100,   101,   103,   109,   104,   105,   106,   107,   108,   102,
        110,   111,   113,   119,   114,   115,   116,   117,   118,   112,
        120,   121,   123,   129,   124,   125,   126,   127,   128,   122,
        130,   131,   133,   139,   134,   135,   136,   137,   138,   132,
        140,   141,   143,   149,   144,   145,   146,   147,   148,   142,
        150,   151,   153,   159,   154,   155,   156,   157,   158,   152,
        260,   261,   263,   269,   264,   265,   266,   267,   268,   262,
        270,   271,   273,   279,   274,   275,   276,   277,   278,   272,
        280,   281,   283,   289,   284,   285,   286,   287,   288,   282,
        290,   291,   293,   299,   294,   295,   296,   297,   298,   292,
        360,   361,   363,   369,   364,   365,   366,   367,   368,   362,
        370,   371,   373,   379,   374,   375,   376,   377,   378,   372,
        380,   381,   383,   389,   384,   385,   386,   387,   388,   382,
        390,   391,   393,   399,   394,   395,   396,   397,   398,   392,
        400,   401,   403,   409,   404,   405,   406,   407,   408,   402,
        300,   301,   303,   309,   304,   305,   306,   307,   308,   302,
        310,   311,   313,   319,   314,   315,   316,   317,   318,   312,
        320,   321,   323,   329,   324,   325,   326,   327,   328,   322,
        330,   331,   333,   339,   334,   335,   336,   337,   338,   332,
        340,   341,   343,   349,   344,   345,   346,   347,   348,   342,
        350,   351,   353,   359,   354,   355,   356,   357,   358,   352,
        410,   411,   413,   419,   414,   415,   416,   417,   418,   412,
        420,   421,   423,   429,   424,   425,   426,   427,   428,   422,
        430,   431,   433,   439,   434,   435,   436,   437,   438,   432,
        440,   441,   443,   449,   444,   445,   446,   447,   448,   442,
        450,   451,   453,   459,   454,   455,   456,   457,   458,   452,
        460,   461,   463,   469,   464,   465,   466,   467,   468,   462,
        470,   471,   473,   479,   474,   475,   476,   477,   478,   472,
        480,   481,   483,   489,   484,   485,   486,   487,   488,   482,
        490,   491,   493,   499,   494,   495,   496,   497,   498,   492,
        500,   501,   503,   509,   504,   505,   506,   507,   508,   502,
        510,   511,

        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,
    }};

    lm_permute_list_t perm_list_oline_a = {0};
    static lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    static lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                MAT_A_C_SIZE,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }
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
LM_UT_CASE_FUNC(lm_ut_permute_oline_cycle_conversion_512_elem_3)
{
    #undef MAT_SAMPLES
    #undef MAT_A_R_SIZE
    #undef MAT_A_C_SIZE
    #define MAT_SAMPLES     1
    #define MAT_A_R_SIZE    2
    #define MAT_A_C_SIZE    512
    int32_t cnt;
    lm_permute_size_t elem_cnt;
    lm_rtn_t result;
    lm_permute_list_t perm_list_cycle_a = {0};
    lm_permute_elem_t perm_list_elem_oline_a[MAT_SAMPLES][MAT_A_R_SIZE * MAT_A_C_SIZE] = {{
        511,   510,
        509,   508,   507,   506,   505,   504,   503,   502,   501,   500,
        499,   498,   497,   496,   495,   494,   493,   492,   491,   490,
        489,   488,   487,   486,   485,   484,   483,   482,   481,   480,
        479,   478,   477,   476,   475,   474,   473,   472,   471,   470,
        469,   468,   467,   466,   465,   464,   463,   462,   461,   460,
        459,   458,   457,   456,   455,   454,   453,   452,   451,   450,
        449,   448,   447,   446,   445,   444,   443,   442,   441,   440,
        439,   438,   437,   436,   435,   434,   433,   432,   431,   430,
        429,   428,   427,   426,   425,   424,   423,   422,   421,   420,
        419,   418,   417,   416,   415,   414,   413,   412,   411,   410,
        409,   408,   407,   406,   405,   404,   403,   402,   401,   400,
        399,   398,   397,   396,   395,   394,   393,   392,   391,   390,
        389,   388,   387,   386,   385,   384,   383,   382,   381,   380,
        379,   378,   377,   376,   375,   374,   373,   372,   371,   370,
        369,   368,   367,   366,   365,   364,   363,   362,   361,   360,
        359,   358,   357,   356,   355,   354,   353,   352,   351,   350,
        349,   348,   347,   346,   345,   344,   343,   342,   341,   340,
        339,   338,   337,   336,   335,   334,   333,   332,   331,   330,
        329,   328,   327,   326,   325,   324,   323,   322,   321,   320,
        319,   318,   317,   316,   315,   314,   313,   312,   311,   310,
        309,   308,   307,   306,   305,   304,   303,   302,   301,   300,
        299,   298,   297,   296,   295,   294,   293,   292,   291,   290,
        289,   288,   287,   286,   285,   284,   283,   282,   281,   280,
        279,   278,   277,   276,   275,   274,   273,   272,   271,   270,
        269,   268,   267,   266,   265,   264,   263,   262,   261,   260,
        259,   258,   257,   256,   255,   254,   253,   252,   251,   250,
        249,   248,   247,   246,   245,   244,   243,   242,   241,   240,
        239,   238,   237,   236,   235,   234,   233,   232,   231,   230,
        229,   228,   227,   226,   225,   224,   223,   222,   221,   220,
        219,   218,   217,   216,   215,   214,   213,   212,   211,   210,
        209,   208,   207,   206,   205,   204,   203,   202,   201,   200,
        199,   198,   197,   196,   195,   194,   193,   192,   191,   190,
        189,   188,   187,   186,   185,   184,   183,   182,   181,   180,
        179,   178,   177,   176,   175,   174,   173,   172,   171,   170,
        169,   168,   167,   166,   165,   164,   163,   162,   161,   160,
        159,   158,   157,   156,   155,   154,   153,   152,   151,   150,
        149,   148,   147,   146,   145,   144,   143,   142,   141,   140,
        139,   138,   137,   136,   135,   134,   133,   132,   131,   130,
        129,   128,   127,   126,   125,   124,   123,   122,   121,   120,
        119,   118,   117,   116,   115,   114,   113,   112,   111,   110,
        109,   108,   107,   106,   105,   104,   103,   102,   101,   100,
        99,    98,    97,    96,    95,    94,    93,    92,    91,    90,
        89,    88,    87,    86,    85,    84,    83,    82,    81,    80,
        79,    78,    77,    76,    75,    74,    73,    72,    71,    70,
        69,    68,    67,    66,    65,    64,    63,    62,    61,    60,
        59,    58,    57,    56,    55,    54,    53,    52,    51,    50,
        49,    48,    47,    46,    45,    44,    43,    42,    41,    40,
        39,    38,    37,    36,    35,    34,    33,    32,    31,    30,
        29,    28,    27,    26,    25,    24,    23,    22,    21,    20,
        19,    18,    17,    16,    15,    14,    13,    12,    11,    10,
        9,     8,     7,     6,     5,     4,     3,     2,     1,     0,

        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,
    }};

    lm_permute_list_t perm_list_oline_a = {0};
    static lm_permute_elem_t perm_list_elem_oline_in_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};
    static lm_permute_elem_t perm_list_elem_oline_out_a[MAT_A_R_SIZE * MAT_A_C_SIZE] = {0};

    for (cnt = 0; cnt < MAT_SAMPLES; cnt++) {

        memcpy((void *)(perm_list_elem_oline_in_a),
               (void *)(perm_list_elem_oline_a[cnt]),
               sizeof(perm_list_elem_oline_in_a));

        result = lm_permute_set(&perm_list_cycle_a,
                                MAT_A_C_SIZE,
                                perm_list_elem_oline_in_a,
                                sizeof(perm_list_elem_oline_in_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_set(&perm_list_oline_a,
                                0,
                                perm_list_elem_oline_out_a,
                                sizeof(perm_list_elem_oline_out_a) / sizeof(lm_permute_elem_t));
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the one-line notation to cycle notation */
        result = lm_permute_oline_to_cycle(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        /* Convert the cycle notation back to one-line notation */
        result = lm_permute_cycle_to_oline(&perm_list_cycle_a, &perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        for (elem_cnt = 0; elem_cnt < perm_list_oline_a.elem.num / 2; elem_cnt++) {

            LM_UT_ASSERT((perm_list_oline_a.elem.ptr[elem_cnt]
                          == perm_list_elem_oline_a[cnt][elem_cnt]),
                         "Incorrect one line notation");
        }

        result = lm_permute_clr(&perm_list_cycle_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");

        result = lm_permute_clr(&perm_list_oline_a);
        LM_UT_ASSERT((result == LM_ERR_CODE(LM_SUCCESS)), "");
    }
}

static lm_ut_case_t lm_ut_permute_cases[] =
{
    {"lm_ut_permute_set_and_clr", lm_ut_permute_set_and_clr, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_wrong_oline_elem", lm_ut_permute_oline_cycle_conversion_wrong_oline_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_wrong_cycle_elem", lm_ut_permute_oline_cycle_conversion_wrong_cycle_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_1_elem", lm_ut_permute_oline_cycle_conversion_1_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_2_elem", lm_ut_permute_oline_cycle_conversion_2_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_3_elem", lm_ut_permute_oline_cycle_conversion_3_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_5_elem", lm_ut_permute_oline_cycle_conversion_5_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_6_elem", lm_ut_permute_oline_cycle_conversion_6_elem, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_512_elem_1", lm_ut_permute_oline_cycle_conversion_512_elem_1, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_512_elem_2", lm_ut_permute_oline_cycle_conversion_512_elem_2, NULL, NULL, 0, 0},
    {"lm_ut_permute_oline_cycle_conversion_512_elem_3", lm_ut_permute_oline_cycle_conversion_512_elem_3, NULL, NULL, 0, 0},
};

static lm_ut_suite_t lm_ut_permute_suites[] =
{
    {"lm_ut_permute_suites", lm_ut_permute_cases, sizeof(lm_ut_permute_cases) / sizeof(lm_ut_case_t), 0, 0}
};

static lm_ut_suite_list_t lm_ut_list[] =
{
    {lm_ut_permute_suites, sizeof(lm_ut_permute_suites) / sizeof(lm_ut_suite_t), 0, 0}
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
int32_t lm_ut_run_permute()
{
    lm_ut_run(lm_ut_list);

    return 0;
}


/*
 *******************************************************************************
 * Private functions
 *******************************************************************************
 */

