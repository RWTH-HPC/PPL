/**


This file implements the Thread Pool and barriers.


*/
#define _GNU_SOURCE 1
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string>
#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <array>
#include <atomic>
#include <unordered_map>
#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>
#include "PThreadsLib.hxx"
#include "cuda_pool_lib.hxx"
#include <cuda.h>
#include <cuda_runtime.h>

std::vector<Thread> *gpuPool;

void setGPUPool(std::vector<Thread> * pool) {
    gpuPool = pool;
}

std::vector<Thread> * getGPUPool() {
    return gpuPool;
}

/**
    Initialization of the pthreads setup:
    1. Set the affinity to the cores of the threads spawned. (Core 0 spawns the first thread and the main thread)
    2. Spawn the thread pool.
    3. Fill the map of thread ids.
*/
void startGPUExecution() {
    pthread_t thread;

    thread = pthread_self();

    for(unsigned int cid = 0; cid < gpuPool->size() ;cid++) {

        gpuPool->at(cid).setResult(pthread_create( gpuPool->at(cid).getThread(), NULL, execute, (void *) &gpuPool->at(cid)));

        if (pthread_mutex_init(gpuPool->at(cid).getQueueLock(), NULL) != 0) {
            std::cout << "Mutex failed to initialize in thread:" << cid << std::endl;
        }
        if (gpuPool->at(cid).getResult() != 0) {
            std::cout << "Failed to initialize thread:" << cid << std::endl;
        }

        auto f = [cid]() {
            cudaSetDevice(cid);
        };

        gpuPool->at(cid).addWork(f);
    }

    //CPU_FREE(cpuset);
}

/**
    Finializing function of the thread-pool and clean up.
*/
void finishGPUExecution() {
    //std::cout << "Finalize" << std::endl;
    for(int i = 0; i < gpuPool->size() ;i++) {
        gpuPool->at(i).setConcluding(true);
    }
    for(int i = 0; i < gpuPool->size() ;i++) {
        pthread_join( *gpuPool->at(i).getThread(), NULL);
        //std::cout << "Thread" << i << " joined" << std::endl;
    }
}


/**
    Synchronizes a GPU with the Host.
*/
void cuda_sync_device(boost::shared_ptr<Bit_Mask> mask) {
    //std::cout << "Barrier" << std::endl;
    mask->isMain();
    mask->activate();
    for(int i = 0; i < mask->size(); i++) {
        if(mask->get(i) == false) {
            std::function<void()> f = [mask]() { thread_Barrier(mask); };
            //std::cout << i << std::endl;
            gpuPool->at(i).addWork(f);
        }
    }
    pthread_barrier_wait (mask->getBarrier());
    mask->isNotMain();
}

