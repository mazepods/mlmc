/*
   These are similar to the Feynman-Kac tests
   for the JUQ paper with Francisco Bernal
*/

#include "mlmc_test.h"
#include "rng.h"

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
// some declarations
//

int option;  // parameter to specify the option (sub-sampling, GM offset)
             // 1 -- sub-sampling, no offset
             // 2 -- nothing
             // 3 -- sub-sampling, offset

static inline void next_inc(int D, float h, float* dW) {
  for (int d=0; d<D; d++) dW[d] = sqrt(h)*next_normal();
}

static inline void step(int D, float h, double& t, float* dW, double* X) {
  for (int d=0; d<D; d++) X[d] += dW[d];
  t += h;
}

static inline void inside(int D, float h, double t, float T, double*X, int& in) {
  double b = 1.0;
  if (option==3) b = b - 0.5826*sqrt(h);

  for (int d=0; d<D; d++) in = in && (fabs(X[d])<=b);
  in = in && (t<T);
}

int main(int argc, char **argv) {

  int N0   = 200; // initial samples on each coarse level
  int Lmin = 2;   // minimum refinement level
  int Lmax = 10;  // maximum refinement level
 
  int   N, L;
  char  filename[32];
  FILE *fp;

#ifdef _OPENMP
  double wtime = omp_get_wtime();
#endif

//
// loop over different options
//

  for (option=1; option<4; option++) {
    // initialise generator, with separate storage for each
    // thread when compiled for OpenMP
#pragma omp parallel
    rng_initialisation();

    sprintf(filename,"fk1_%d.txt",option);
    fp = fopen(filename,"w");

    if (option<=3) {
      if (option==1)
        printf("\n ---- option %d: EM with split ----\n",option);
      else if (option==2)
        printf("\n ---- option %d: EM without split ----\n",option);
      else if (option==3)
        printf("\n ---- option %d: EM with split and GM offset ----\n",option);

      if (option<3) {
        N = 50000;    // samples for convergence tests
        L = 6;        // levels for convergence tests 
      }
      else {
        N = 500000;   // samples for convergence tests, more because error is small
        L = 6;        // levels for convergence tests 
      }
      //      float Eps[] = { 0.001, 0.002, 0.005, 0.01, 0.02, 0.0 };
      float val = NAN;
      float Eps[] = { 0.0005, 0.001, 0.002, 0.005, 0.01};
      int size_eps = NELEMS(Eps);

      complexity_test(N,L,N0,Eps,size_eps,Lmin,Lmax,fp);

    fclose(fp);

    // print out time taken, if using OpenMP
#ifdef _OPENMP
    printf(" execution time = %f s\n",omp_get_wtime() - wtime);
    wtime = omp_get_wtime();
#endif

    // delete generator and any associated storage
#pragma omp parallel
    rng_termination();

//
// now do 100 MLMC calcs in parallel
//
#pragma omp parallel
    rng_initialisation();

    sprintf(filename, "fk1.txt");
    fp = fopen(filename,"w");
    mlmc_test_n(val,100,N0,Eps,size_eps,Lmin,Lmax,fp);

    fclose(fp);

#ifdef _OPENMP
    printf(" execution time = %f s\n",omp_get_wtime() - wtime);
#endif

#pragma omp parallel
    rng_termination();

    }
  }
}


/*-------------------------------------------------------
%
% level l estimator
%
*/


void mlmc_l(int l, int N, double *sums) {

  int    M=4, D=3, split;
  float  T, h0, hf, hc;

  if (option==1 || option==3) {
    //  split = 4 * (1<<l);
    split = 1<<l;
  }
  else if (option==2) {
    split = 1;
  }

  T  = 1.0;
  h0 = 0.1;

  hf = h0 / powf((float)M,(float) l);
  hc = h0 / powf((float)M,(float) (l-1));

  for (int k=0; k<6; k++) sums[k] = 0.0;

  /*
  OpenMP reduction of C++ array sections is discussed here:
  https://www.openmp.org/spec-html/5.0/openmpsu107.html
  and an example is given on page 301 here:
  https://www.openmp.org/wp-content/uploads/openmp-examples-5.0.0.pdf
  */
  
#pragma omp parallel for shared(M,D,split, T,h0,hf,hc) reduction(+:sums[0:7])
  for (int np = 0; np<N; np++) {
    // variables declared here inside OpenMP parallel loop
    // will have local allocation for each thread
    int    in_f, in_c, set_f, set_c;
    float  Pf, Pc, dP, dWf[3], dWc[3];
    double tf, tc, t_split, Xf[3], Xc[3], X_split[3];

    int RNG_count = 0;  

    for (int d=0; d<D; d++) {
      Xf[d] = 0.0;
      Xc[d] = 0.0;
    }

    tf   = 0.0;
    tc   = 0.0;
    in_f = 1;
    in_c = 1;
    set_f = 0;
    set_c = 0;

    Pf = 0.0;
    Pc = 0.0;

    // level 0

    if (l==0) {
      do {
        next_inc(D,hf,dWf);
	RNG_count++;
	
        step(D,hf,tf,dWf,Xf);

        inside(D,hf,tf,T,Xf,in_f);

      } while (in_f);

      Pf = tf;
    }

    // level l>0

    else {
      do {
        for (int d=0; d<D; d++) dWc[d] = 0.0;

        for (int m=0; m<M; m++) {
          next_inc(D,hf,dWf);
          RNG_count++;

          for (int d=0; d<D; d++) dWc[d] += dWf[d];

          step(D,hf,tf,dWf,Xf);

          inside(D,hf,tf,T,Xf,in_f);

          if ( (!in_f) && (!set_f) ) {
            set_f = 1;
            Pf    = tf;
          }
	}

        step(D,hc,tc,dWc,Xc);

        inside(D,hc,tc,T,Xc,in_c);

        if ( (!in_c) && (!set_c) ) {
          set_c = 1;
          Pc    = tc;
        }

      } while (in_f && in_c);

    // split continuation paths 

      if (in_f) {
        t_split = tf;
        for (int d=0; d<D; d++) X_split[d] = Xf[d];

        for (int s=0; s<split; s++) {
	  // reset state at split
          in_f = 1;
          tf = t_split;
          for (int d=0; d<D; d++) Xf[d] = X_split[d];

	  // continue until exit
          do {
            next_inc(D,hf,dWf);
            RNG_count++;

            step(D,hf,tf,dWf,Xf);

            inside(D,hf,tf,T,Xf,in_f);

          } while (in_f);

          Pf += tf / ((float) split);
        }
      }

      if (in_c) {
        t_split = tc;
        for (int d=0; d<D; d++) X_split[d] = Xc[d];

        for (int s=0; s<split; s++) {
	  // reset state at split
          in_c = 1;
          tc = t_split;
          for (int d=0; d<D; d++) Xc[d] = X_split[d];

	  // continue until exit
          do {
            next_inc(D,hc,dWc);
            RNG_count++;

            step(D,hc,tc,dWc,Xc);

            inside(D,hc,tc,T,Xc,in_c);

          } while (in_c);

          Pc += tc / ((float) split);
        }
      }
    }

    dP = Pf-Pc;

    sums[0] += D*RNG_count; // add number of RNG calls
    sums[1] += dP;
    sums[2] += dP*dP;
    sums[3] += dP*dP*dP;
    sums[4] += dP*dP*dP*dP;
    sums[5] += Pf;
    sums[6] += Pf*Pf;
  }

}
