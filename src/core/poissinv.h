/////////////////////////////////////////////////////////////////
//                                                             //
// This software was written by Mike Giles, 2014               //
//                                                             //
// It is copyright University of Oxford, and provided under    //
// the terms of the GNU GPLv3 license:                         //
// http://www.gnu.org/licenses/gpl.html                        //
//                                                             //
// Commercial users wanting to use the software under a more   //
// permissive license, such as BSD, should contact the author: //
// mike.giles@maths.ox.ac.uk                                   //
//                                                             //
/////////////////////////////////////////////////////////////////

#include <math.h>

// declare prototype for inverse Normal CDF function
// defined at the bottom of this header file

double normcdfinv_as241(double);

//
// This double precision function computes the inverse
// of the Poisson CDF
//
// u   = CDF value in range (0,1)
// lam = Poisson rate
//
// For lam < 1e15,  max |error| no more than 1
//  ave |error| < 1e-16*max(4,lam) for lam < 1e9
//              < 1e-6             for lam < 1e15
//
// For lam > 1e15, the errors will be about 1 ulp.
//


// As described in the TOMS paper, there are two versions;
// the first is optimised for MIMD execution, whereas
// the second is designed for vector execution

double poissinv_core(double, double, double);

double poissinv(double U, double Lam);

double poisscinv(double V, double Lam);

inline double poissinv_core(double U, double V, double Lam); 

double poissinv_v(double U, double Lam); 

//////////////////////////////////////////////////////////////////////
//                                                                  //
// The routine below is a C version of the code in                  //
//                                                                  //
// ALGORITHM AS241: APPLIED STATS (1988) VOL. 37, NO. 3, 477-44.    //
// http://lib.stat.cmu.edu/apstat/241                               //
//                                                                  //
// The relative error is less than 1e-15, and the accuracy is       //
// verified in the accompanying MATLAB code as241.m                 //
//                                                                  //
//////////////////////////////////////////////////////////////////////

double normcdfinv_as241(double p);

