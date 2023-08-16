/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/

#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>
#include "PThreadsLib.hxx"

#ifndef CUDA_POOL_LIB_H
#define CUDA_POOL_LIB_H


void setGPUPool(std::vector<Thread> * pool);
std::vector<Thread> * getGPUPool();


void startGPUExecution();

/**
    Finializing function of the thread-pool and clean up.
*/
void finishGPUExecution();

/**
    Implements the synchronization of a GPU
*/
void cuda_sync_device(boost::shared_ptr<Bit_Mask> mask);


#endif
