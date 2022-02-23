# C compiler
CC = gcc

# Compile flags
CFLAGS = -Wall -g -O2 -Wfloat-equal -Wunused-result -Wsign-compare -Wmaybe-uninitialized
#CFLAGS += -Wconversion

# For Cortex-M4 with HW FPU
#CFLAGS += -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard

# Test coverage
#CFLAGS += -fprofile-arcs -ftest-coverage

# Profiling option:
# GCC manual: -pg = generate extra code to write profile information suitable for the analysis program prof (for -p) or gprof (for -pg).
#CFLAGS += -pg

# Sanitize
#CFLAGS += -fsanitize=address,leak,undefined,shift,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,bounds-strict,alignment,object-size,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr

# Includes
INCLUDES = -Isrc/ -Iinc/ -Itest/

# Linker flags
LFLAGS = 

# For Cortex-M4
#LFLAGS += -specs=nosys.specs

# Link libraries
LIBS = -lm

# Lin Matrix library source files
LM_SRCS = \
src/lm_log.c \
src/lm_assert.c \
src/lm_err.c \
src/lm_mat.c \
src/lm_chk.c \
src/lm_shape.c \
src/lm_permute.c \
src/lm_oper.c \
src/lm_oper_dot.c \
src/lm_oper_norm.c \
src/lm_oper_axpy.c \
src/lm_oper_gemm.c \
src/lm_lu.c \
src/lm_qr.c \
src/lm_symm_hess.c \
src/lm_symm_eigen.c

# Lin Matrix unit test source files
UT_SRCS = \
test/lm_ut.c \
test/lm_ut_framework.c \
test/lm_ut_mat.c \
test/lm_ut_chk.c \
test/lm_ut_shape.c \
test/lm_ut_oper.c \
test/lm_ut_oper_dot.c \
test/lm_ut_oper_norm.c \
test/lm_ut_oper_axpy.c \
test/lm_ut_oper_gemm.c \
test/lm_ut_permute.c \
test/lm_ut_lu.c \
test/lm_ut_qr.c \
test/lm_ut_symm_hess.c \
test/lm_ut_symm_eigen.c

# Lin Matrix example source files
EXAMPLE1_SRCS = examples/lm_mat_mult/lm_mat_mult.c
EXAMPLE2_SRCS = examples/lm_qr_decomp/lm_qr_decomp.c
EXAMPLE3_SRCS = examples/lm_ellipsoid_fit/lm_ellipsoid_fit.c
EXAMPLE4_SRCS = examples/lm_perf_measure/lm_perf_measure.c

# Lin Matrix library objects
LM_OBJS = $(LM_SRCS:.c=.o) 
UT_OBJS = $(UT_SRCS:.c=.o)
EXAMPLE1_OBJS = $(EXAMPLE1_SRCS:.c=.o) 
EXAMPLE2_OBJS = $(EXAMPLE2_SRCS:.c=.o) 
EXAMPLE3_OBJS = $(EXAMPLE3_SRCS:.c=.o)
EXAMPLE4_OBJS = $(EXAMPLE4_SRCS:.c=.o) 

# Dependency
DEPENDS = $(LM_SRCS:.c=.d) $(UT_SRCS:.c=.d) $(EXAMPLE1_SRCS:.c=.d) $(EXAMPLE2_SRCS:.c=.d) $(EXAMPLE3_SRCS:.c=.d) $(EXAMPLE4_SRCS:.c=.d)

# Pattern rule ‘%.o : %.c’ says how to make any file xxx.o from another file xxx.c.
%.o : %.c
		$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Output files
LM_LIB = lm_lib.a
LM_UT = lm_ut
LM_EXAMPLE1 = lm_mat_mult
LM_EXAMPLE2 = lm_qr_decomp
LM_EXAMPLE3 = lm_ellipsoid_fit
LM_EXAMPLE4 = lm_perf_measure

.PHONY: clean

all:      $(LM_UT) $(LM_EXAMPLE1) $(LM_EXAMPLE2) $(LM_EXAMPLE3) $(LM_EXAMPLE4)

-include $(DEPENDS)
		
$(LM_LIB): $(LM_OBJS)
		ar rc $@ $^ && ranlib $@
		@echo $(LM_LIB) has been compiled

$(LM_UT): $(UT_OBJS) $(LM_LIB)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(LM_UT) $(UT_OBJS) $(LFLAGS) $(LM_LIB) $(LIBS) 
		@echo $(LM_UT) has been compiled
        
$(LM_EXAMPLE1): $(EXAMPLE1_OBJS) $(LM_LIB)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(LM_EXAMPLE1) $(EXAMPLE1_OBJS) $(LFLAGS) $(LM_LIB) $(LIBS) 
		@echo $(LM_EXAMPLE1) has been compiled
        
$(LM_EXAMPLE2): $(EXAMPLE2_OBJS) $(LM_LIB)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(LM_EXAMPLE2) $(EXAMPLE2_OBJS) $(LFLAGS) $(LM_LIB) $(LIBS)
		@echo $(LM_EXAMPLE2) has been compiled
        
$(LM_EXAMPLE3): $(EXAMPLE3_OBJS) $(LM_LIB)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(LM_EXAMPLE3) $(EXAMPLE3_OBJS) $(LFLAGS) $(LM_LIB) $(LIBS)
		@echo $(LM_EXAMPLE3) has been compiled
       
$(LM_EXAMPLE4): $(EXAMPLE4_OBJS) $(LM_LIB)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(LM_EXAMPLE4) $(EXAMPLE4_OBJS) $(LFLAGS) $(LM_LIB) $(LIBS)
		@echo $(LM_EXAMPLE4) has been compiled
       
RM_LIST = $(DEPENDS) *.gcov gmon.out \
		$(LM_OBJS) $(LM_LIB) $(LM_SRCS:.c=.gcda) $(LM_SRCS:.c=.gcno) \
		$(LM_UT) $(LM_UT).exe $(UT_OBJS) $(UT_SRCS:.c=.gcda) $(UT_SRCS:.c=.gcno) \
		$(LM_EXAMPLE1) $(LM_EXAMPLE1).exe  $(EXAMPLE1_OBJS) $(EXAMPLE1_SRCS:.c=.gcda) $(EXAMPLE1_SRCS:.c=.gcno) \
		$(LM_EXAMPLE2) $(LM_EXAMPLE2).exe  $(EXAMPLE2_OBJS) $(EXAMPLE2_SRCS:.c=.gcda) $(EXAMPLE2_SRCS:.c=.gcno) \
		$(LM_EXAMPLE3) $(LM_EXAMPLE3).exe  $(EXAMPLE3_OBJS) $(EXAMPLE3_SRCS:.c=.gcda) $(EXAMPLE3_SRCS:.c=.gcno) \
		$(LM_EXAMPLE4) $(LM_EXAMPLE4).exe  $(EXAMPLE4_OBJS) $(EXAMPLE4_SRCS:.c=.gcda) $(EXAMPLE4_SRCS:.c=.gcno)

clean:
		$(RM) $(RM_LIST)

