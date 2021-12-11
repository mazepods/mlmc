/*
    MLMC tests for the Reflected Diffusions paper with Kavita Ramanan
    using a Milstein discretisation with adaptive timestepping
    when we approach the boundaries.

    NOTE: needs variables to be double precision because of very
    small timesteps on finest levels
*/

#include "mlmc_test.h"  // master MLMC file
#include "mlmc_rng.cpp"   // new file with RNG functions

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// some application-specific bits

int D = 3;   // dimension
int option;

void reflected_l(int, int, double *);

void drift_dt(double *x, double *v){
  double d = 0.0;
  for (int i=0; i<D; i++) d += x[i]*x[i];
  d = fmax(1.0,sqrt(d));

  // penalisation drift
  for (int i=0; i<D; i++) v[i] = x[i]*(1.0-d)/d;
}

double dt(double *x, int l){
  double d = 0.0;
  for (int i=0; i<D; i++) d += x[i]*x[i];
  d = fmax(0.0, 1.0-sqrt(d));

  return fmin( pow(0.5,  (double) (l+3)),
         fmax( pow(0.25, (double) (l+3)),
	       0.0625 * d*d) );
}

void reflect(double *x){

  if (option==3) return;
  
  double d = 0;
  for (int i=0; i<D; i++) d += x[i]*x[i];

  if (d > 1.0) {
    if (option==1)
      d = 2.0/sqrt(d) - 1.0;   // reflection scheme
    else
      d = 1.0/sqrt(d);         // projection scheme

    for (int i=0; i<D; i++) x[i] = d*x[i];
  }
}


// main code

int main(int argc, char **argv) {

  int N    = 10000;  // samples for convergence tests
  int L    = 8;      // levels for convergence tests
  
  int N0   = 100;    // initial samples on each level
  int Lmin = 2;      // minimum refinement level
  int Lmax = 12;     // maximum refinement level
 
  float Eps[] = { 0.0002, 0.0005, 0.001, 0.002, 0.005 };
  int size_eps = NELEMS(Eps);
  char filename[32];
  FILE *fp;

#ifdef _OPENMP
  double wtime = omp_get_wtime();
#endif

//
// loop over different options
//

  for (option=1; option<4; option++) {

    // initialise generator and storage for each thread
#pragma omp parallel
    rng_initialisation();

    sprintf(filename,"reflected_%d.txt",option);
    fp = fopen(filename,"w");

    if (option==1) {
      printf("\n ---- option %d: reflection scheme ----\n",option);
    }
    else if (option==2) {
      printf("\n ---- option %d: projection scheme ----\n",option);
    }
    else {
      printf("\n ---- option %d: penalisation scheme ----\n",option);
    }
    
    complexity_test(N,L,N0,Eps,size_eps,Lmin,Lmax,fp);

    fclose(fp);
    
    // print out time taken
#ifdef _OPENMP
    printf(" execution time = %f s\n",omp_get_wtime() - wtime);
#endif

    // delete generator and storage
#pragma omp parallel
    rng_termination();
  }
}


//-------------------------------------------------------
//
//  level l multiple thread OpenMP estimator
//

void mlmc_l(int l, int N, double *sums) {

  for (int k=0; k<7; k++) sums[k] = 0.0;

  /* branch depending on whether or not level 0 */
  if (l==0) {

#pragma omp parallel for reduction(+:sums[0:7])
    for (int n2=0; n2<N; n2++) {
      double v[3], xf[3],xc[3], dW[3],dWf[3],dWc[3];
int   nrng = 0; double T  = 1.0; double t  = 0.0; double tf = 0.0; double tc = 0.0; 
      double hf = 0.0;
      double hc = 0.0;

      for (int i=0; i<D; i++) {
	xf[i]  = 0.0;
        xc[i]  = 0.0;
        dWf[i] = 0.0;
        dWc[i] = 0.0;
      }

      while (t<T) {
        t = tf;

	if (t > 0.0f) {
          for (int i=0; i<D; i++) dWf[i] = sqrt(hf)*next_normal();
          nrng +=1;

	  drift_dt(xf,v);
          for (int i=0; i<D; i++) xf[i] += v[i] + dWf[i];
          reflect(xf);
	}

	hf = fmin(dt(xf,l), T-tf);
	tf = tf+hf;
      }
	
      double Pf = 0.0;
      for (int i=0; i<D; i++) Pf += xf[i]*xf[i];

      sums[0] += nrng;     // add number of Normal vectors as cost
      sums[1] += Pf;
      sums[2] += Pf*Pf;
      sums[3] += Pf*Pf*Pf;
      sums[4] += Pf*Pf*Pf*Pf;
      sums[5] += Pf;
      sums[6] += Pf*Pf;
    }
  }
   
  else {

#pragma omp parallel for reduction(+:sums[0:7])
    for (int n2=0; n2<N; n2++) {
      double v[3], xf[3],xc[3], dW[3],dWf[3],dWc[3];

      int nrng = 0;

      double T  = 1.0;
      double t  = 0.0;
      double tm = 0.0;
      double tf = 0.0;
      double tc = 0.0;

      double hf = 0.0;
      double hc = 0.0;

      for (int i=0; i<D; i++) {
	xf[i]  = 0.0;
        xc[i]  = 0.0;
        dWf[i] = 0.0;
        dWc[i] = 0.0;
      }

      while (t<T) {
	tm = t;
	t  = fmin(tf,tc);

        for (int i=0; i<D; i++) {
          dW[i] = sqrt(t-tm)*next_normal();
          dWf[i] += dW[i];
          dWc[i] += dW[i];
        }
        nrng +=1;

        if (t==tf) {
	  drift_dt(xf,v);
          for (int i=0; i<D; i++) xf[i] += v[i] + dWf[i];
          reflect(xf);
          hf = fmin(dt(xf,l), T-tf);
          tf = tf+hf;
          for (int i=0; i<D; i++) dWf[i] = 0.0;
	}
	
        if (t==tc) {
	  drift_dt(xc,v);
          for (int i=0; i<D; i++) xc[i] += v[i] + dWc[i];
          reflect(xc);
          hc = fmin(dt(xc,l-1), T-tc);
          tc = tc+hc;
          for (int i=0; i<D; i++) dWc[i] = 0.0;
	}
	
      }
	
      double Pf = 0.0;
      for (int i=0; i<D; i++) Pf = Pf + xf[i]*xf[i];
      double dP = 0.0;
      for (int i=0; i<D; i++) dP = dP + xf[i]*xf[i] - xc[i]*xc[i];

      sums[0] += nrng;     // add number of Normal vectors as cost
      sums[1] += dP;
      sums[2] += dP*dP;
      sums[3] += dP*dP*dP;
      sums[4] += dP*dP*dP*dP;
      sums[5] += Pf;
      sums[6] += Pf*Pf;
    }
  }
}
