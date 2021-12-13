/*
    This tests the use of multilevel Monte Carlo for an SDE
    which requires adaptive time-stepping

    Developed with James Whittle, summer 2014

    M.B. Giles, C. Lester, J. Whittle.
   'Non-nested adaptive timesteps in multilevel Monte Carlo computations'.
    Monte Carlo and Quasi-Monte Carlo Methods 2014, Springer, 2015.
*/
  

#include "rng.h"
#include "mlmc_test.h"

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// some application-specific bits

int D = 2;  // dimension

//
//------ drift function ---------
//

void drift(double *q, double beta, double *v) {
  double q2 = 0.0;
  for (int i=0; i<D; i++) q2 += q[i]*q[i];
  for (int i=0; i<D; i++) v[i] = -(4.0*beta/(1.0-q2)) * q[i];
}

//
//------ adaptive timestep ---------
//

double dt(double *q, double beta, int l) {
  double q2 = 0.0;
  for (int i=0; i<D; i++) q2 += q[i]*q[i];
  q2 = 1.0 - sqrt(q2);
  double h = pow(0.5,(double) l) * q2*q2 / fmax(2.0*beta,50.0);
  return h;
}
//
//------ clamp ---------
//

void clamp(double *q, double beta) {
  double q2 = 0.0;
  for (int i=0; i<D; i++) q2 += q[i]*q[i];
  for (int i=0; i<D; i++) 
    q[i] = q[i] * fmin(1.0, (1.0-1.0e-5)/(1.0e-100 + sqrt(q2)));
}

//
//------ main code ------------------------
//

int main(int argc, char **argv) {

  int N    = 10000;  // samples for convergence tests
  int L    = 6;      // levels for convergence tests
  
  int N0   = 100;    // initial samples on each level
  int Lmin = 2;      // minimum refinement level
  int Lmax = 10;     // maximum refinement level
 
  float val = NAN;
  float Eps[] = { 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01 };
  int size_eps = NELEMS(Eps);
  char filename[32];
  FILE *fp;

#ifdef _OPENMP
  double wtime = omp_get_wtime();
#endif

  // initialise generator and storage for each thread
#pragma omp parallel
  rng_initialisation();

  sprintf(filename,"adapted.txt");
  fp = fopen(filename,"w");

  complexity_test(N,L,N0,Eps,size_eps,Lmin,Lmax,fp);

  fclose(fp);
    
  // print out time taken
#ifdef _OPENMP
  printf(" execution time = %f s\n",omp_get_wtime() - wtime);
#endif

  // delete generator and storage
#pragma omp parallel
  rng_termination();

  //
  // 100 MLMC calcs
  //

#pragma omp parallel
  rng_initialisation();

  sprintf(filename, "adapted_100.txt");
  fp = fopen(filename,"w");
  mlmc_test_n(val,100,N0,Eps,size_eps,Lmin,Lmax,fp);

  fclose(fp);

    // print out time taken, if using OpenMP
#ifdef _OPENMP
    printf(" execution time = %f s\n",omp_get_wtime() - wtime);
    wtime = omp_get_wtime();
#endif

    // delete generator and storage
#pragma omp parallel
    rng_termination();

}
  

//-------------------------------------------------------
//
//  level l multiple thread OpenMP estimator
//

void mlmc_l(int l, int N, double *sums) {

  for (int k=0; k<7; k++) sums[k] = 0.0;

  double beta = 4.0, T = 1.0;

#pragma omp parallel for reduction(+:sums[0:7])
  for (int n2=0; n2<N; n2++) {
    double v[2], qf[2],qc[2], dWf[2],dWc[2];

    double t  = 0.0;
    double tm = 0.0;
    double tf = 0.0;
    double tc = 0.0;

    double hf = 0.0;
    double hc = 0.0;

    double cost = 0.0;

    for (int i=0; i<D; i++) {
      qf[i]  = 0.0;
      qc[i]  = 0.0;
      dWf[i] = 0.0;
      dWc[i] = 0.0;
    }

  /* branch depending on whether or not level 0 */
    if (l==0) {

      while (t<T) {
        t = tf;
        drift(qf,beta,v);
        for (int i=0; i<D; i++) {
   	  double dW = sqrt(hf)*next_normal();
          qf[i] = qf[i] + v[i]*hf + 2.0*dW;
        }
        clamp(qf,beta);
        hf = fmin( dt(qf,beta,l), T-tf);
        tf = fmin(tf+hf, T);

	cost += 1.0;
      }
    }
 
    else {

      while (t<T) {
        tm = t;
        t  = fmin(tf,tc);

        for (int i=0; i<D; i++) {
          double dW = sqrt(t-tm)*next_normal();
          dWf[i] += dW;
          dWc[i] += dW;
	}

        if (tf==t) {
	  drift(qf,beta,v);
          for (int i=0; i<D; i++) {
            qf[i]  = qf[i] + v[i]*hf + 2.0*dWf[i];
            dWf[i] = 0.0;
	  }
          clamp(qf,beta);
          hf = fmin(dt(qf,beta,l), T-tf);
          tf = fmin(tf+hf, T);
        }

        if (tc==t) {
	  drift(qc,beta,v);
          for (int i=0; i<D; i++) {
            qc[i]  = qc[i] + v[i]*hc + 2*dWc[i];
            dWc[i] = 0.0;
	  }
          clamp(qc,beta);
          hc = fmin(dt(qc,beta,l-1), T-tc);
          tc = fmin(tc+hc, T);
        }

	cost += 1.0;
      }
    }

    double Pf = 0.0;
    for (int i=0; i<D; i++) Pf = Pf + qf[i]*qf[i];
    double dP = 0.0;
    for (int i=0; i<D; i++) dP = dP + qf[i]*qf[i] - qc[i]*qc[i];

    sums[0] += cost;     // add number of Normal vectors as cost
    sums[1] += dP;
    sums[2] += dP*dP;
    sums[3] += dP*dP*dP;
    sums[4] += dP*dP*dP*dP;
    sums[5] += Pf;
    sums[6] += Pf*Pf;
  }
}

