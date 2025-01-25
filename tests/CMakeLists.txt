cmake_minimum_required (VERSION 2.6)
project(hnsw LANGUAGES CXX)
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )


add_executable(nns dosearch.cpp
	graphdiverse.hpp
	../src/nnsearch.hpp
	../src/metrics.hpp
    ../src/iomanager.hpp)
