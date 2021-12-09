/*
   P = mlmc(Lmin,Lmax,N0,eps, mlmc_l, alpha,beta,gamma, Nl,Cl)
 
   multilevel Monte Carlo control routine

   Lmin  = minimum level of refinement       >= 2
   Lmax  = maximum level of refinement       >= Lmin
   N0    = initial number of samples         > 0
   eps   = desired accuracy (rms error)      > 0 
 
   mlmc_l(l,N,sums)   low-level function
        l       = level
        N       = number of paths
        sums[0] = sum(cost)
        sums[1] = sum(Y)
        sums[2] = sum(Y^2)
        where Y are iid samples with expected value:
        E[P_0]           on level 0
        E[P_l - P_{l-1}] on level l>0

   alpha -> weak error is  O(2^{-alpha*l})
   beta  -> variance is    O(2^{-beta*l})
   gamma -> sample cost is O(2^{gamma*l})

   if alpha, beta, gamma are not positive then they will be estimated

   P   = value
   Nl  = number of samples at each level
   Cl  = average cost of samples at each level
*/


#include <math.h>
#include <stdio.h>

#define ALPHA 0
#define BETA 1
#define GAMMA 2

void regression(int, float *, float *, float &a, float &b);
float estimate(int L, float *z, int type, float min =0.0f);
void update_samples(int L, int *dNl, float *Vl, float *Cl, float eps, float theta, double *Nl);

float mlmc(int Lmin, int Lmax, int N0, float eps,
           void (*mlmc_l)(int, int, double *),
           int *Nl, float *Cl,
           float alpha_0 = 0.0f, float beta_0 = 0.0f, float gamma_0 = 0.0f) {

  double sums[7];
  double *suml[3];
  for (int i=0; i<3; i++) suml[i] = (double *)malloc((Lmax+1)*sizeof(double));

  float *ml   = (float *)malloc((Lmax+1)*sizeof(float)),
        *Vl   = (float *)malloc((Lmax+1)*sizeof(float)),
        *NlCl = (float *)malloc((Lmax+1)*sizeof(float));
  float alpha, beta, gamma, sum, theta;
  int   *dNl = (int *)malloc((Lmax+1)*sizeof(int));
  int   L, converged;

  int   diag = 0;  // diagnostics, set to 0 for none

  //
  // check input parameters
  //

  if (Lmin<2) {
    fprintf(stderr,"error: needs Lmin >= 2 \n");
    exit(1);
  }
  if (Lmax<Lmin) {
    fprintf(stderr,"error: needs Lmax >= Lmin \n");
    exit(1);
  }

  if (N0<=0 || eps<=0.0f) {
    fprintf(stderr,"error: needs N>0, eps>0 \n");
    exit(1);
  }

  //
  // initialisation
  //

  alpha = fmax(0.0f,alpha_0);
  beta  = fmax(0.0f,beta_0);
  gamma = fmax(0.0f,gamma_0);
  theta = 0.25f;             // MSE split between bias^2 and variance

  L = Lmin;
  converged = 0;

  for(int l=0; l<=Lmax; l++) {
    Nl[l]   = 0;
    Cl[l]   = powf(2.0f,(float)l*gamma);
    NlCl[l] = 0.0f;

    for(int n=0; n<3; n++) suml[n][l] = 0.0;
  }

  for(int l=0; l<=Lmin; l++) dNl[l] = N0;

  //
  // main loop
  //

  while (!converged) {

    //
    // update sample sums
    //

    for (int l=0; l<=L; l++) {
      if (diag) printf(" %d ",dNl[l]);

      if (dNl[l]>0) {
        for(int n=0; n<7; n++) sums[n] = 0.0;
        mlmc_l(l,dNl[l],sums);
        suml[0][l] += (float) dNl[l];
        suml[1][l] += sums[1];
        suml[2][l] += sums[2];
        NlCl[l]    += sums[0];  // sum total cost
      }
    }
    if (diag) printf(" \n");

    //
    // compute absolute average, variance and cost,
    // correct for possible under-sampling,
    // and set optimal number of new samples
    //

    //sum = 0.0f;

    for (int l=0; l<=L; l++) {
      ml[l] = fabs(suml[1][l]/suml[0][l]);
      Vl[l] = fmaxf(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0.0f);
      if (gamma_0 <= 0.0f) Cl[l] = NlCl[l] / suml[0][l];

      if (l>1) {
        ml[l] = fmaxf(ml[l],  0.5f*ml[l-1]/powf(2.0f,alpha));
        Vl[l] = fmaxf(Vl[l],  0.5f*Vl[l-1]/powf(2.0f,beta));
      }
    }

    update_samples(L, dNl, Vl, Cl, eps, theta, suml[0]);
 
    //
    // use linear regression to estimate alpha, beta, gamma if not given
    //

    if (alpha_0 <= 0.0f) {
      alpha = estimate(L,ml,ALPHA,0.5f);
      if (diag) printf(" alpha = %f \n",alpha);
    }

    if (beta_0 <= 0.0f) {
      beta = estimate(L,Vl,BETA,0.5f);
      if (diag) printf(" beta = %f \n",beta);
    }

     if (gamma_0 <= 0.0f) {
      gamma = estimate(L,Cl,GAMMA,0.5f);
      if (diag) printf(" gamma = %f \n",gamma);
    }

    //
    // if (almost) converged, estimate remaining error and decide 
    // whether a new level is required
    //

    sum = 0.0;
    for (int l=0; l<=L; l++)
      sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][l]);

    if (sum==0) {
      if (diag) printf(" achieved variance target \n");

      converged = 1;
      float rem = ml[L] / (powf(2.0f,alpha)-1.0f);

      if (rem > sqrtf(theta)*eps) {
        if (L==Lmax)
          printf("*** failed to achieve weak convergence *** \n");
        else {
          converged = 0;
          L++;
          Vl[L] = Vl[L-1]/powf(2.0f,beta);
          Cl[L] = Cl[L-1]*powf(2.0f,gamma);

          if (diag) printf(" L = %d \n",L);

          update_samples(L, dNl, Vl, Cl, eps, theta, suml[0]);
        }
      }
    }
  }

  //
  // finally, evaluate multilevel estimator and set outputs
  //

  float P = 0.0f;
  for (int l=0; l<=L; l++) {
    P    += suml[1][l]/suml[0][l];
    Nl[l] = suml[0][l];
    Cl[l] = NlCl[l] / Nl[l];
  }

  return P;
}

//
// linear regression routine
//

void regression(int N, float *x, float *y, float &a, float &b){

  float sum0=0.0f, sum1=0.0f, sum2=0.0f, sumy0=0.0f, sumy1=0.0f;

  for (int i=0; i<N; i++) {
    sum0  += 1.0f;
    sum1  += x[i];
    sum2  += x[i]*x[i];

    sumy0 += y[i];
    sumy1 += y[i]*x[i];
  }

  a = (sum0*sumy1 - sum1*sumy0) / (sum0*sum2 - sum1*sum1);
  b = (sum2*sumy0 - sum1*sumy1) / (sum0*sum2 - sum1*sum1);
}

//
// use linear regression routine to estimate alpha, beta, gamma
//

float estimate(int L, float *z, int type, float min) {
  float var, sum;
  int a = 1;
  float *x = (float *)malloc((L+1)*sizeof(float));
  float *y = (float *)malloc((L+1)*sizeof(float));

  if (type == GAMMA) a = -1;

  for (int l=1; l<=L; l++) {
    x[l-1] = l;
    y[l-1] = a * log2f(x[l]);
  }
  regression(L,x,y,var,sum);
  return fmax(var,min);
}

//
// gives the number of additional samples which will need to be generated in the next pass 
//

void update_samples(int L, int *dNl, float *Vl, float *Cl, float eps, float theta, double *Nl) {
  float sum = 0.0f;
  for (int l=0; l<=L; l++) sum += sqrtf(Vl[l]*Cl[l]);
  for (int l=0; l<=L; l++)
  dNl[l] = ceilf(fmaxf(0.0f, sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)-Nl[l]));
}
