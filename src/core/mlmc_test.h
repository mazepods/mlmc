#ifndef MLMC_TEST_H
#define MLMC_TEST_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

// https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
// variadic macro to print to both file and stdout
#define PRINTF2(fp, ...) {printf(__VA_ARGS__);fprintf(fp,__VA_ARGS__);}

#define NELEMS(a) ((int)(sizeof(a)/sizeof(*a)))

void complexity_test(int N, int L, int N0, float *Eps, int size_eps, int Lmin, int Lmax, FILE *fp);

void mlmc_test_n(float val, int n, int N0, float *Eps, int size_eps, int Lmin, int Lmax, FILE *fp);

#endif
