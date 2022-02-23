#!/bin/bash
lcov -c -o ut_coverage.info -d src/
genhtml ut_coverage.info -o ut_coverage
