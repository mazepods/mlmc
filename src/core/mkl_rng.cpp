#include "rng.h"
//------- MKL/VSL random number generation when using OpenMP -----------

#ifdef _OPENMP

#include <mkl.h>
#include <mkl_vsl.h>
#include <memory.h>
#include <omp.h>

/* each OpenMP thread has its own VSL RNG and storage */

#define NRV 16384  // number of random variables
VSLStreamStatePtr stream;
float *uniforms,      *normals,      *exponentials;
int    uniforms_count, normals_count, exponentials_count;
#pragma omp threadprivate(stream, uniforms,uniforms_count, \
        normals,normals_count, exponentials,exponentials_count)

//
// RNG routines
//

void rng_initialisation(){
  int tid = omp_get_thread_num();
  vslNewStream(&stream, VSL_BRNG_MRG32K3A,1337);
  long long skip = ((long long) (tid+1)) << 48;
  vslSkipAheadStream(stream,skip);
  uniforms     = (float *)malloc(NRV*sizeof(float));
  normals      = (float *)malloc(NRV*sizeof(float));
  exponentials = (float *)malloc(NRV*sizeof(float));
  uniforms_count     = 0;  // this means there are no random
  normals_count      = 0;  // numbers in the arrays currently
  exponentials_count = 0;  // 
}

void rng_termination(){
  vslDeleteStream(&stream);
  free(uniforms);
  free(normals);
  free(exponentials);
}

float next_uniform(){
  if (uniforms_count==0) {
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                  stream,NRV,uniforms,0.0f,1.0f);
    uniforms_count = NRV;
  }
  return uniforms[--uniforms_count];
}

float next_normal(){
  if (normals_count==0) {
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                  stream,NRV,normals,0.0f,1.0f);
    normals_count = NRV;
  }
  return normals[--normals_count];
}

float next_exponential(){
  if (exponentials_count==0) {
    vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE,
                     stream,NRV,exponentials,0.0f,1.0f);
    exponentials_count = NRV;
  }
  return exponentials[--exponentials_count];
}

#endif

