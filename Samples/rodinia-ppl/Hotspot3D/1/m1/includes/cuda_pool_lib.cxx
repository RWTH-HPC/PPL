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
#include "cuda_pool_lib.hxx"
#include <cuda.h>
#include <cuda_runtime.h>
#include "BitMask.hxx"

std::vector<ThreadGPU> *gpuPool;

void setGPUPool(std::vector<ThreadGPU> * pool) {
    gpuPool = pool;
}

std::vector<ThreadGPU> * getGPUPool() {
    return gpuPool;
}

// returns the pointer to the pthreads thread object which is necessary for initializing and finalizing a thread.
    pthread_t * ThreadGPU::getThread() { return &thread;}

    // returns the pthreads thread object( currently not used, might be removed in the future).
    pthread_t ThreadGPU::getThreadReal() { return thread;}

    // returns the pointer to the mutex for locking the queue (currently not used outside of the data structure, might be removed in the future).
    pthread_mutex_t *ThreadGPU::getQueueLock() {return &queueLock;}

    // sets the error value for creating a thread in pthreads. (might be extended for handling errors)
    void ThreadGPU::setResult(int res) { result = res;}

    // returns the error value for creating a thread in pthreads.
    int ThreadGPU::getResult() {return result;}

    // set the value of concluding, necessary before the thread can join the main program flow again. (might be simplified -> only setting concluding to true).
    void ThreadGPU::setConcluding(bool con) {concluding = con;}

    // returns the current value of concluding.
    bool ThreadGPU::getConcluding() {return concluding;}

    // adds a task to the ring buffer, if the buffer is not full.
    void ThreadGPU::addWork(std::function<void()> task) {
        pthread_mutex_lock(&queueLock);
        if(size == BUFFER_SIZE) {
            std::cout << "Queue full" << std::endl;
        } else if (size != 0) {
            tail = (tail + 1) % BUFFER_SIZE;
        }
        workBuffer.at(tail) = task;
        size++;
        pthread_mutex_unlock(&queueLock);
    }

    // returns the current first task in the buffer, without removing it.
    std::function<void()> ThreadGPU::getTask() {
        return workBuffer.at(head);
    }

    // removes the current first task from the buffer.
    void ThreadGPU::removeFinishedJob() {
        pthread_mutex_lock(&queueLock);
        if (size == 0) {
            std::cout << "Queue is empty" <<std::endl;
        }
        if (size > 1) {
            head = (head + 1) % BUFFER_SIZE;
        }
        size--;
        pthread_mutex_unlock(&queueLock);
    }

    // returns true if the buffer is empty.
    bool ThreadGPU::isEmpty() {
        bool res = false;
        pthread_mutex_lock(&queueLock);
        if (size == 0) {
            res = true;
        }
        pthread_mutex_unlock(&queueLock);
        return res;
    }

    // constructor to initialize all necessary variables.
    ThreadGPU::ThreadGPU() {
        std::vector<std::function<void()>> init(BUFFER_SIZE);
        workBuffer = init;
        queueLock = PTHREAD_MUTEX_INITIALIZER;
        head = 0;
        tail = 0;
        size = 0;
        result = 0;
        concluding = false;
    }


/*
    Implementation of the executor which executes the jobs from the work queue until the queue is empty and the thread should return.
*/
void *executeGPU(void *ptr) {
ThreadGPU *worker;
worker = (ThreadGPU *) ptr;
    while(!worker->getConcluding() || !worker->isEmpty()) {
        if ( !worker->isEmpty()) {
            std::function<void()> exec = worker->getTask();
            exec();
            worker->removeFinishedJob();
        }
    }
    pthread_exit(NULL);
}


/**
    Initialization of the pthreads setup:
    1. Set the affinity to the cores of the threads spawned. (Core 0 spawns the first thread and the main thread)
    2. Spawn the thread pool.
    3. Fill the map of thread ids.
*/
void startGPUExecution() {
    int s;
    pthread_t thread;

    cpu_set_t cpuset;// = CPU_ALLOC(N);

    thread = pthread_self();

    for(unsigned int cid = 0; cid < gpuPool->size() ;cid++) {

        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);

        gpuPool->at(cid).setResult(pthread_create( gpuPool->at(cid).getThread(), NULL, executeGPU, (void *) &gpuPool->at(cid)));

        s = pthread_setaffinity_np(gpuPool->at(cid).getThreadReal(), sizeof(cpu_set_t), &cpuset);
                if (s != 0)
                    handle_error_en(s, "pthread_setaffinity_np");

                s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
                if (s != 0)
                    handle_error_en(s, "pthread_getaffinity_np");

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

