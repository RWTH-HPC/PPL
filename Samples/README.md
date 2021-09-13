# Samples
This directory contains various sample codes for the PPL as well as C implementations using global optimizations written by hand.

## clusters

This directory contains two cluster descriptions which can be used with the PPL tool. These descriptions specify a subset of two nodes from the [CLAIX 2018 MPI and CLAIX 2018 GPU](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/fbd107191cf14c4b8307f44f545cf68a/) cluster respectively.

## Hello World 

Contains the first Hello World example written in the PPL.

## Rodinia PPL

The translation of the Rodinia OpenMP benchmarks into the PPL.

## cholesky

Implementation of global optimizations targeting the cholesky decomposition algorithm only a C implementation is available so far.

## classification

Implementation of a batch classification algorithm written in both the PPL and a C implementation with and without global optimizations.

## jacobi 

Implementation of a jacobi solver algorithm written in both the PPL and a C implementation with and without global optimizations.

## kmeans

Implementation of global optimizations targeting the kmeans algorithm. A PPL version can be found in the Rodinia-PPL directory.

## monte_carlo 

Implementation of a monte carlo estimation algorithm written in both the PPL and a C implementation with and without global optimizations.

## multi-filter 

Implementation of a multi filter convolutuion written in both the PPL and a C implementation with and without global optimizations.

## nn 

Implementation of a neural network forward pass written in both the PPL and a C implementation with and without global optimizations.
