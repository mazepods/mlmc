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

#ifndef MLMC_H
#define MLMC_H

#define ALPHA 0
#define BETA 1
#define GAMMA 2

//
// linear regression routine
//
void regression(int N, float *x, float *y, float &a, float &b);

//
// use linear regression routine to estimate alpha, beta, gamma
//
float estimate(int L, float *z, int type, float min =0.0f);

//
// gives the number of additional samples which will need to be generated in the next pass
//
void update_samples(int L, int *dNl, float *Vl, float *Cl, float eps, float theta, double *Nl);

void mlmc_l(int l, int N, double *sums);

float mlmc(int Lmin, int Lmax, int N0, float eps, int *Nl, float *Cl,
           float alpha_0 = 0.0f, float beta_0 = 0.0f, float gamma_0 = 0.0f);

#endif
