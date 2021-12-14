ifeq ($(OS),Windows_NT)
	CC := clang:
	CFLAGS := -g -Wall -std=c++11
	INC := -I. -I.. -Isrc/core
	LIB := -lm
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CC := g++
		CFLAGS := -g -Wall -std=c++11
		INC := -I. -I.. -Isrc/core -I/usr/include/x86_64-linux-gnu/c++/9/
		LIB := -lm
	endif
	ifeq ($(UNAME_S),Darwin)
		CC := clang
		CFLAGS := -g -Wall -std=c++11
		INC := -I. -I.. -Isrc/core -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
		LIB := -lm -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
	endif
endif

CCOMP	:= icpc
COMPFLAGS := -O2 -no-prec-div -xHost -qopenmp -qopt-report -mkl -static-intel

SRCDIR  := src
BINDIR  := bin
OBJDIR  := obj

VPATH := src:src/core

STOBJS := $(patsubst $(SRCDIR)/core/%.cpp, $(OBJDIR)/st/%.o, $(wildcard $(SRCDIR)/core/*.cpp))
MTOBJS := $(patsubst $(SRCDIR)/core/%.cpp, $(OBJDIR)/mt/%.o, $(wildcard $(SRCDIR)/core/*.cpp))
STSRCS := $(filter-out src/timing_omp.cpp, ${wildcard $(SRCDIR)/*.cpp})
STBINS := $(patsubst $(SRCDIR)/%.cpp, %, $(STSRCS))
MTBINS := timing_omp mcqmc06_omp 

all: st mt

st: $(STBINS)

mt: $(MTBINS)

$(OBJDIR)/st/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) $(LIB)

$(OBJDIR)/mt/%.o: %.cpp
	$(CCOMP) $(COMPFLAGS) -c $< -o $@ $(INC) $(LIB)

mcqmc06: $(OBJDIR)/st/mcqmc06.o $(STOBJS) rng.h
	$(CC) $(CFLAGS) $(OBJDIR)/st/random_rng.o $(OBJDIR)/st/mlmc.o $(OBJDIR)/st/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

mcqmc06_omp: $(OBJDIR)/mt/mcqmc06.o $(MTOBJS)
	$(CCOMP) $(COMPFLAGS) $(OBJDIR)/mt/mkl_rng.o $(OBJDIR)/mt/mlmc.o $(OBJDIR)/mt/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

ctmp: $(OBJDIR)/st/ctmp.o $(STOBJS)
	$(CC) $(CFLAGS) $(OBJDIR)/st/poissinv.o $(OBJDIR)/st/random_rng.o $(OBJDIR)/st/mlmc.o $(OBJDIR)/st/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

nested: $(OBJDIR)/st/nested.o $(STOBJS)
	$(CC) $(CFLAGS) $(OBJDIR)/st/random_rng.o $(OBJDIR)/st/mlmc.o $(OBJDIR)/st/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

nested_omp: $(OBJDIR)/mt/nested.o $(MTOBJS)
	$(CCOMP) $(COMPFLAGS) $(OBJDIR)/mt/mkl_rng.o $(OBJDIR)/mt/mlmc.o $(OBJDIR)/mt/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

adapted: $(OBJDIR)/st/adapted.o $(STOBJS)
	$(CC) $(CFLAGS) $(OBJDIR)/st/random_rng.o $(OBJDIR)/st/mlmc.o $(OBJDIR)/st/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

adapted_omp: $(OBJDIR)/mt/adapted.o $(MTOBJS)
	$(CCOMP) $(COMPFLAGS) $(OBJDIR)/mt/mkl_rng.o $(OBJDIR)/mt/mlmc.o $(OBJDIR)/mt/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

reflected: $(OBJDIR)/st/reflected.o $(STOBJS)
	$(CC) $(CFLAGS) $(OBJDIR)/st/random_rng.o $(OBJDIR)/st/mlmc.o $(OBJDIR)/st/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

reflected_omp: $(OBJDIR)/mt/reflected.o $(MTOBJS)
	$(CCOMP) $(COMPFLAGS) $(OBJDIR)/mt/mkl_rng.o $(OBJDIR)/mt/mlmc.o $(OBJDIR)/mt/mlmc_test.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

timing: $(OBJDIR)/st/timing.o $(STOBJS)
	$(CC) $(CFLAGS) $(OBJDIR)/st/random_rng.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

timing_omp: $(OBJDIR)/mt/timing_omp.o $(MTOBJS)
	$(CCOMP) $(COMPFLAGS) $(OBJDIR)/mt/mkl_rng.o $< -o $(BINDIR)/$@ $(INC) $(LIB)

clean:
	$(RM) -rf $(BINDIR)/* $(OBJDIR)/*
	mkdir -p $(OBJDIR)/st
	mkdir -p $(OBJDIR)/mt
