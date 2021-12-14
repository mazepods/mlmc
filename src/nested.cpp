/*
%
% Takashi Goda's model EVPPI problem for our paper
%
*/

#include "mlmc_test.h"
#include "rng.h"

#include <omp.h>

//
// main code
//

int main(int argc, char **argv) {
  
  int N    = 200000; // samples for convergence tests
  int L    = 10;     // levels for convergence tests 

  int N0   = 1000;   // initial samples on each level
  int Lmin = 2;      // minimum refinement level
  int Lmax = 20;     // maximum refinement level
 
  float val = NAN;
  float Eps[] = { 0.0001, 0.0002, 0.0005, 0.001, 0.002 };
  int size_eps = NELEMS(Eps);

  FILE *fp;
  char filename[32];

//
// main MLMC calculation
// 
#ifdef _OPENMP
  double wtime = omp_get_wtime();
#endif

#pragma omp parallel
  rng_initialisation();

  sprintf(filename, "nested.txt");
  fp = fopen(filename,"w");

  complexity_test(N,L,N0,Eps,size_eps,Lmin,Lmax,fp);

  fclose(fp);

#ifdef _OPENMP
  printf(" execution time = %f s\n",omp_get_wtime() - wtime);
#endif

#pragma omp parallel
  rng_termination();
  
//
// now do 100 MLMC calcs in parallel
//
#pragma omp parallel
  rng_initialisation();

  sprintf(filename, "nested_100.txt");
  fp = fopen(filename,"w");
  mlmc_test_n(val,100,N0,Eps,size_eps,Lmin,Lmax,fp);

  fclose(fp);

#ifdef _OPENMP
    printf(" execution time = %f s\n",omp_get_wtime() - wtime);
    wtime = omp_get_wtime();
#endif

#pragma omp parallel
    rng_termination();

}


/*-------------------------------------------------------
%
% level l estimator
%
*/


void mlmc_l(int l, int N, double *sums) {

  float Pf, Pc, dP, X, Y, f2;

  for (int m=0; m<7; m++) sums[m]=0.0;
    
  int nf = 1<<l;
  int nc = nf/2;

#pragma omp parallel for shared(nf,nc) reduction(+:sums[0:7])
  for (int nn=0; nn<N; nn++) {

    // level 0

    if (l==0) {
      Pf = 0;
      dP = Pf;
    }

    // level l>0, with antithetic sampler
    else {
      X  = next_normal();  // outer variable

      float sum1=0.0, sum2=0.0, sum3=0.0;
      for (int n=0; n<nc; n++) {
        Y  = next_normal(); // inner variables
        f2 = X + Y;
        sum1 += f2;
	sum3 += fmaxf(0.0,f2);
        Y  = next_normal(); // inner variables
        f2 = X + Y;
        sum2 += f2;
	sum3 += fmaxf(0.0,f2);
      }
      
      float ave = sum3/nf;
      Pf  = ave - fmaxf(0.0,(sum1+sum2)/nf);
      Pc  = ave - 0.5*(fmaxf(0.0,sum1/nc) + fmaxf(0.0,sum2/nc));
      dP = Pf - Pc;
    }

    sums[0] += nf;   // cost
    sums[1] += dP;
    sums[2] += dP*dP;
    sums[3] += dP*dP*dP;
    sums[4] += dP*dP*dP*dP;
    sums[5] += Pf;
    sums[6] += Pf*Pf;
  }
}
