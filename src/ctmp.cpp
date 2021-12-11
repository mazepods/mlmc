/*
%
% model stochastic reaction testcase from paper by Anderson and Higham
%
*/

#include <random>           // C++11 random number generators
#include <functional>

#include "mlmc_test.h"
#include "poissinv.h"

void ctmp_l(int, int, double *);

// declare generator and output distribution

std::default_random_engine rng;
std::uniform_real_distribution<float> uniform(0.0f,1.0f);
auto next_uniform = std::bind(std::ref(uniform), std::ref(rng));

//
// main code
//

int main(int argc, char **argv) {
  
  int N    = 100000; // samples for convergence tests
  int L    = 6;      // levels for convergence tests 

  int N0   = 100;    // initial samples on each level
  int Lmin = 2;      // minimum refinement level
  int Lmax = 6;      // maximum refinement level
 
  float Eps[] = { 1.0, 2.0, 5.0, 20.0, 0.0 };

  FILE *fp;

//
// main MLMC calculation
// 

  rng.seed(1234);
  uniform.reset();

  fp = fopen("ctmp.txt","w");
  complexity_test(N,L,N0,Eps,Lmin,Lmax,fp);
  fclose(fp);
  
//
// now do 100 MLMC calcs in parallel
//

  /*
  float val = 0.0;
  fp = fopen("ctmp_100.txt","w");
  mlmc_test_100(val, N0,Eps,Lmin,Lmax, fp);
  fclose(fp);
  */
}


/*-------------------------------------------------------
%
% level l estimator
%
*/

void propensity(float *q, float *lam){
  lam[0] = 25.0;
  lam[1] = 1000.0*q[0];
  lam[2] = 0.001*q[1]*(q[1]-1.0);
  lam[3] = 0.1*q[0];
  lam[4] = q[1];
}


void mlmc_l(int l, int N, double *sums) {

  float u, p1, p2;
  int   n_states=3, n_reactions=5;
  float nu[3][5] = { { 1,  0,  0, -1,  0 },  // stoichiometric matrix
                     { 0,  1, -2,  0, -1 },
                     { 0,  0,  1,  0,  0 } };

  for (int m=0; m<7; m++) sums[m]=0.0;

  int M = 4;
  int nf = 8<<(2*l);
  int nc = nf/M;

  float hf = 1.0/((float) nf);
  float hc = 1.0/((float) nc);
  
  for (int nn=0; nn<N; nn++) {

    float qf[] = {0.0, 0.0, 0.0};
    float qc[] = {0.0, 0.0, 0.0};

    float lamf[5], lamc[5];
    
    // level 0

    if (l==0) {
      for (int n=0; n<nf; n++) {
	propensity(qf,lamf);
	for (int r=0; r<n_reactions; r++) {
	  u  = next_uniform();
          p1 = poissinv(u,hf*lamf[r]);
	  for (int s=0; s<n_states; s++) {
            qf[s] = qf[s] + p1*nu[s][r];
	  }
	}
        for (int s=0; s<n_states; s++) qf[s] = fmaxf(0.0f,qf[s]);
      }
    }

    // level l>0
    else {
      for (int n=0; n<nc; n++) {
	propensity(qc,lamc);
	
        for (int m=0; m<M; m++) {
          propensity(qf,lamf);
	  for (int r=0; r<n_reactions; r++) {
            u  = next_uniform();
            p1 = poissinv(u,hf*fminf(lamf[r],lamc[r]));
	    u  = next_uniform();
            p2 = poissinv(u,hf*fabsf(lamf[r]-lamc[r]));

            for (int s=0; s<n_states; s++) {
	      if (lamf[r]<lamc[r]) {
                qf[s] = qf[s] +  p1    *nu[s][r];
                qc[s] = qc[s] + (p1+p2)*nu[s][r];
	      }
	      else {
                qf[s] = qf[s] + (p1+p2)*nu[s][r];
                qc[s] = qc[s] +  p1    *nu[s][r];
	      }
            }
	  }
          for (int s=0; s<n_states; s++) qf[s] = fmaxf(0.0f,qf[s]);
	}
        for (int s=0; s<n_states; s++) qc[s] = fmaxf(0.0f,qc[s]);
      }
    }

    float Pf = qf[2];
    float Pc = qc[2];
    float dP = Pf-Pc;
    
    sums[0] += nf;   // cost
    sums[1] += dP;
    sums[2] += dP*dP;
    sums[3] += dP*dP*dP;
    sums[4] += dP*dP*dP*dP;
    sums[5] += Pf;
    sums[6] += Pf*Pf;
  }
}
