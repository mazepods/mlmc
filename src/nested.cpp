/*
%
% Takashi Goda's model EVPPI problem for our paper
%
*/

//#include "mlmc_test.cpp"     // master MLMC file
#include "mlmc_test_100.cpp" // master file for 100 tests
#include <random>           // C++11 random number generators
#include <functional>

void nested_l(int, int, double *);

// declare generator and output distribution

std::default_random_engine rng;
std::normal_distribution<float> normal(0.0f,1.0f);
auto next_normal = std::bind(std::ref(normal), std::ref(rng));

//
// main code
//

int main(int argc, char **argv) {
  
  int N    = 200000; // samples for convergence tests
  int L    = 10;     // levels for convergence tests 

  int N0   = 1000;   // initial samples on each level
  int Lmin = 2;      // minimum refinement level
  int Lmax = 20;     // maximum refinement level
 
  float Eps[] = { 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.0 };

  FILE *fp;

//
// main MLMC calculation
// 

  rng.seed(1234);
  normal.reset();

  fp = fopen("nested.txt","w");
  mlmc_test(nested_l, N,L, N0,Eps, Lmin,Lmax, fp);
  fclose(fp);
  
//
// now do 100 MLMC calcs in parallel
//

  /*
  float val = 0.0;
  fp = fopen("nested_100.txt","w");
  mlmc_test_100(mcqmc06_l, val, N0,Eps,Lmin,Lmax, fp);
  fclose(fp);
  */
}


/*-------------------------------------------------------
%
% level l estimator
%
*/


void nested_l(int l, int N, double *sums) {

  float Pf, Pc, dP, X, Y, f2;

  for (int m=0; m<7; m++) sums[m]=0.0;
    
  int nf = 1<<l;
  int nc = nf/2;

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
