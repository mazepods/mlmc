/*

RNG performance test

*/

#include "rng.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
// main code
//

int main(int argc, char **argv) {

  int maxt = omp_get_max_threads();
  
  for (int nt = 1; nt<=maxt; nt++) {
    omp_set_num_threads(nt);

  // create generator and any associated storage
#pragma omp parallel
    rng_initialisation();

    double wtime = omp_get_wtime();
    double sums[2]={0,0};
    int    N = 1<<30;

#pragma omp parallel for reduction(+:sums[0:2])
    for (int n = 0; n<N; n++) {
      float z = next_normal();
      sums[0] += z;
      sums[1] += z*z;
    }

    double mean = sums[0] /((double) N);
    double sd   = sqrt(sums[1] /((double) N) - mean*mean);
    wtime = omp_get_wtime() - wtime;
  
    printf(" mean = %f, sd = %f \n", mean, sd);
    printf(" nthreads = %d, RNGs/sec = %g \n", nt, (double) N / wtime);

    // delete generator and any associated storage
#pragma omp parallel
    rng_termination();
  }
}

