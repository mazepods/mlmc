#include "rng.h"

//----- C++11 random number generation when not using OpenMP -------------

#ifndef _OPENMP

#include <random>           // C++11 random number generators
#include <functional>

// declare generator and output distributions

std::default_random_engine rng;
std::uniform_real_distribution<float> uniform(0.0f,1.0f);
std::normal_distribution<float> normal(0.0f,1.0f);
std::exponential_distribution<float> exponential(1.0f);

float next_uniform() { return uniform(rng); }
float next_normal() { return normal(rng); }
float next_exponential() { return exponential(rng); }

void rng_initialisation() {
    rng.seed(1234);
    uniform.reset();
    normal.reset();
    exponential.reset();
}

void rng_termination() {
}

#endif
