/*

RNG performance test

*/

#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

//
// main code
//

int main(int argc, char **argv) {

  rng_initialisation();

  clock_t t=clock();
  double  sums[2]={0,0};
  int     N = 1<<27;

  for (int n = 0; n<N; n++) {
    float z = next_normal();
    sums[0] += z;
    sums[1] += z*z;
  }

  double mean = sums[0] /((double) N);
  double sd   = sqrt(sums[1] /((double) N) - mean*mean);
  t = clock() - t;
  float time = ((float) t)/CLOCKS_PER_SEC;
  
  printf(" mean = %f, sd = %f \n", mean, sd);
  printf(" RNGs/sec = %g \n", (float) N / time);
}

