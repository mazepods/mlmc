/*
% These are similar to the MLMC tests for the MCQMC06 paper
% using a Milstein discretisation with 2^l timesteps on level l
%
% The figures are slightly different due to
% -- change in MSE split
% -- change in cost calculation
% -- different random number generation
% -- switch to S_0=100
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

int option;  // parameter to specify the financial option

float ncff(float x){
  return 0.5f+0.5f*erff(sqrtf(0.5f)*x);
}

//
// main code
//

int main(int argc, char **argv) {

  int N0   = 200;  // initial samples on each level
  int Lmin = 2;    // minimum refinement level
  int Lmax = 10;   // maximum refinement level
 
  int   N, L;
  float Eps[11];
  int size_eps;
  char  filename[32];
  FILE *fp;

#ifdef _OPENMP
  double wtime = omp_get_wtime();
#endif

//
// loop over different payoffs
//

  for (option=1; option<6; option++) {
    // initialise generator, with separate storage for each
    // thread when compiled for OpenMP
#pragma omp parallel
    rng_initialisation();

    sprintf(filename,"mcqmc06_%d.txt",option);
    fp = fopen(filename,"w");

    if (option==1) {
      printf("\n ---- option %d: European call ----\n",option);
      N      = 20000;    // samples for convergence tests
      L      = 8;        // levels for convergence tests 
      float Eps2[] = { 0.005, 0.01, 0.02, 0.05, 0.1 };
      memcpy(Eps,Eps2,sizeof(Eps2));
      size_eps = NELEMS(Eps2);
    }
    else if (option==2) {
      printf("\n ---- option %d: Asian call ----\n",option);
      N      = 20000;    // samples for convergence tests
      L      = 8;        // levels for convergence tests 
      float Eps2[] = { 0.005, 0.01, 0.02, 0.05, 0.1 };
      memcpy(Eps,Eps2,sizeof(Eps2));
      size_eps = NELEMS(Eps2);
    }
    else if (option==3) {
      printf("\n ---- option %d: lookback call ----\n",option);
      N      = 20000;    // samples for convergence tests
      L      = 10;       // levels for convergence tests 
      float Eps2[] = { 0.005, 0.01, 0.02, 0.05, 0.1 };
      memcpy(Eps,Eps2,sizeof(Eps2));
      size_eps = NELEMS(Eps2);
    }
    else if (option==4) {
      printf("\n ---- option %d: digital call ----\n",option);
      N      = 200000;   // samples for convergence tests
      L      = 8;        // levels for convergence tests 
      float Eps2[] = { 0.01, 0.02, 0.05, 0.1, 0.2 };
      memcpy(Eps,Eps2,sizeof(Eps2));
      size_eps = NELEMS(Eps2);
    }
    else if (option==5) {
      printf("\n ---- option %d: barrier call ----\n",option);
      N      = 200000;   // samples for convergence tests
      L      = 8;        // levels for convergence tests 
      float Eps2[] = { 0.005, 0.01, 0.02, 0.05, 0.1 };
      memcpy(Eps,Eps2,sizeof(Eps2));
      size_eps = NELEMS(Eps2);
    }

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
    // print exact analytic value, based on S0=K
    //

    float T, r, sig, sig2, K, B, d1, d2, val, k, d3, d4;

    T   = 1.0f;
    r   = 0.05f;
    sig = 0.2f;
    sig2 = sig*sig;
    K   = 100.0f;
    B   = 0.85f*K;

    d1  = (r+0.5f*sig2)*T / (sig*sqrtf(T));
    d2  = (r-0.5f*sig2)*T / (sig*sqrtf(T));

    if (option==1)
      val = K*( ncff(d1) - expf(-r*T)*ncff(d2) );
    else if (option==2)
      val = nanf("");
    else if (option==3) {
      k   = 0.5f*sig2/r;
      val = K*( ncff(d1) - ncff(-d1)*k - exp(-r*T)*(ncff(d2) - ncff(d2)*k) );
    }
    else if (option==4)
      val = K*expf(-r*T)*ncff(d2);
    else {
      k   = 0.5f*sig2/r;
      d3  = (2.0f*logf(B/K) + (r+0.5f*sig2)*T) / (sig*sqrtf(T));
      d4  = (2.0f*logf(B/K) + (r-0.5f*sig2)*T) / (sig*sqrtf(T));
      val = K*(                         ncff(d1) - expf(-r*T)*ncff(d2)
   - powf(K/B,1.0f-1.0f/k)*((B*B)/(K*K)*ncff(d3) - expf(-r*T)*ncff(d4) ) );
    }

    //
    // now do 100 MLMC calcs
    //

    // initialise generator and storage for each thread
#pragma omp parallel
    rng_initialisation();

    sprintf(filename,"mcqmc06_%d_100.txt",option);
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
}


/*-------------------------------------------------------
%
% level l estimator
%
*/

void mlmc_l(int l, int N, double *sums) {

  // variables declared here are shared by all OpenMP threads
  int   nf, nc;
  float K, T, r, sig, B, hf, hc;

  // model parameters

  K   = 100.0f;
  T   = 1.0f;
  r   = 0.05f;
  sig = 0.2f;
  B   = 0.85f*K;

  nf = 1<<l;
  nc = nf/2;

  hf = T / ((float) nf);
  hc = T / ((float) nc);

  for (int k=0; k<7; k++) sums[k] = 0.0;

  /*
  OpenMP reduction of C++ array sections is discussed here:
  https://www.openmp.org/spec-html/5.0/openmpsu107.html
  and an example is given on page 301 here:
  https://www.openmp.org/wp-content/uploads/openmp-examples-5.0.0.pdf
  */
  
#pragma omp parallel for shared(K,T,r,sig,B,nf,nc) reduction(+:sums[0:7])
  for (int np = 0; np<N; np++) {
    // variables declared here inside OpenMP parallel loop
    // will have local allocation for each thread
    float X0, Xf, Xc, Af, Ac, Mf, Mc, Bf, Bc,
          Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP;
    float dWf[2], dIf[2], Lf[2];
    
    X0 = K;

    Xf = X0;
    Xc = Xf;

    Af  = 0.5f*hf*Xf;
    Ac  = 0.5f*hc*Xc;

    Mf  = Xf;
    Mc  = Xc;

    Bf  = 1.0f;
    Bc  = 1.0f;

    if (l==0) {
      dWf[0] = sqrt(hf) * next_normal();
      Lf[0]  = - next_exponential();
      dIf[0] = sqrt(hf/12.0f)*hf * next_normal();

      Xf0 = Xf;
      Xf  = Xf + r*Xf*hf + sig*Xf*dWf[0]
               + 0.5f*sig*sig*Xf*(dWf[0]*dWf[0]-hf);
      vf  = sig*Xf0;
      Af  = Af + 0.5f*hf*Xf + vf*dIf[0];
      Mf  = fminf(Mf,
            0.5f*(Xf0+Xf-sqrtf((Xf-Xf0)*(Xf-Xf0)-2.0f*hf*vf*vf*Lf[0])));
      Bf  = Bf*(1.0f-expf(-2.0f*fmaxf(0.0f,(Xf0-B)*(Xf-B)/(hf*vf*vf))));
    }

    else {
      for (int n=0; n<nc; n++) {
        dWf[0] = sqrt(hf) * next_normal();
        dWf[1] = sqrt(hf) * next_normal();
        Lf[0]  = - next_exponential();
        Lf[1]  = - next_exponential();
        dIf[0] = sqrt(hf/12.0f)*hf * next_normal();
        dIf[1] = sqrt(hf/12.0f)*hf * next_normal();

        for (int m=0; m<2; m++) {
          Xf0 = Xf;
          Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
                   + 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
          vf  = sig*Xf0;
          Af  = Af + hf*Xf + vf*dIf[m];
          Mf  = fminf(Mf,
                0.5f*(Xf0+Xf-sqrtf((Xf-Xf0)*(Xf-Xf0)-2.0f*hf*vf*vf*Lf[m])));
          Bf  = Bf*(1.0f-expf(-2.0f*fmaxf(0.0f,(Xf0-B)*(Xf-B)/(hf*vf*vf))));
        }

        dWc = dWf[0] + dWf[1];
        ddW = dWf[0] - dWf[1];

        Xc0 = Xc;
        Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

        vc  = sig*Xc0;
        Ac  = Ac + hc*Xc + vc*(dIf[0]+dIf[1] + 0.25f*hc*ddW);
        Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
        Mc  = fminf(Mc,
            0.5f*(Xc0+Xc1-sqrtf((Xc1-Xc0)*(Xc1-Xc0)-2.0f*hf*vc*vc*Lf[0])));
        Mc  = fminf(Mc,
            0.5f*(Xc1+Xc -sqrtf((Xc -Xc1)*(Xc -Xc1)-2.0f*hf*vc*vc*Lf[1])));
        Bc  = Bc *(1.0f-expf(-2.0f*fmaxf(0.0f,(Xc0-B)*(Xc1-B)/(hf*vc*vc))));
        Bc  = Bc *(1.0f-expf(-2.0f*fmaxf(0.0f,(Xc1-B)*(Xc -B)/(hf*vc*vc))));
      }
      Af = Af - 0.5f*hf*Xf;
      Ac = Ac - 0.5f*hc*Xc;
    }

    if (option==1) {
      Pf  = fmaxf(0.0f,Xf-K);
      Pc  = fmaxf(0.0f,Xc-K);
    }
    else if (option==2) {
      Pf  = fmaxf(0.0f,Af-K);
      Pc  = fmaxf(0.0f,Ac-K);
    }
    else if (option==3) {
      Pf  = Xf - Mf;
      Pc  = Xc - Mc;
    }
    else if (option==4) {
      Pf  = K*ncff((Xf0+r*Xf0*hf-K)/(sig*Xf0*sqrt(hf)));
      if (l==0)
        Pc = Pf;
      else
        Pc = K
           * ncff((Xc0+r*Xc0*hc+sig*Xc0*dWf[0]-K)/(sig*Xc0*sqrt(hf)));
    }
    else if (option==5) {
      Pf  = Bf*fmaxf(0.0f,Xf-K);
      Pc  = Bc*fmaxf(0.0f,Xc-K);
    }

    dP  = exp(-r*T)*(Pf-Pc);
    Pf  = exp(-r*T)*Pf;

    if (l==0) dP = Pf;

    sums[0] += nf;     // add number of timesteps as cost
    sums[1] += dP;
    sums[2] += dP*dP;
    sums[3] += dP*dP*dP;
    sums[4] += dP*dP*dP*dP;
    sums[5] += Pf;
    sums[6] += Pf*Pf;
  }

}

