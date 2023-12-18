/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/

#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>
#include "BitMask.hxx"

#ifndef CUDA_POOL_LIB_H
#define CUDA_POOL_LIB_H

#define BUFFER_SIZE 16384

#define handle_error_en(en, msg) \
       do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)


/**
* Definition of the thread class containing all information regarding a single thread within the thread pool.
* This implementation contains the thread object from pthreads, as well as the result of its generation for potential error handling.
* Additionally, the parameter workBuffer implements a ring buffer of fixed size, for storing submitted tasks.
* The parameters head, tail, size and queueLock are all dependent on the ring buffer:
*   head: the first element in the buffer.
*   tail: the last element in the buffer.
*   size: the current number of unfinished tasks within the ring buffer.
*   queueLock: a mutex lock for accessing the ring buffer, to avoid data races.
* The parameter concluding indicates whether the thread is intended to join the main process again once the buffer is empty.
*/
class ThreadGPU {
    private:
    pthread_t thread;
    std::vector<std::function<void()>> workBuffer;
    int head;
    int tail;
    int size;
    int result;
    bool concluding;
    pthread_mutex_t queueLock;

    public:

    // returns the pointer to the pthreads thread object which is necessary for initializing and finalizing a thread.
    pthread_t *getThread();

    // returns the pthreads thread object( currently not used, might be removed in the future).
    pthread_t getThreadReal();

    // returns the pointer to the mutex for locking the queue (currently not used outside of the data structure, might be removed in the future).
    pthread_mutex_t *getQueueLock();

    // sets the error value for creating a thread in pthreads. (might be extended for handling errors)
    void setResult(int res);

    // returns the error value for creating a thread in pthreads.
    int getResult();

    // set the value of concluding, necessary before the thread can join the main program flow again. (might be simplified -> only setting concluding to true).
    void setConcluding(bool con);

    // returns the current value of concluding.
    bool getConcluding();

    // adds a task to the ring buffer, if the buffer is not full.
    void addWork(std::function<void()> task);

    // returns the current first task in the buffer, without removing it.
    std::function<void()> getTask();

    // removes the current first task from the buffer.
    void removeFinishedJob();

    // returns true if the buffer is empty.
    bool isEmpty();

    // constructor to initialize all necessary variables.
    ThreadGPU();

};



void setGPUPool(std::vector<ThreadGPU> * pool);
std::vector<ThreadGPU> * getGPUPool();


/*
    Implementation of the executor which executes the jobs from the work queue until the queue is empty and the thread should return.
*/
void *executeGPU(void *ptr);

/**
    Initialization of the pthreads setup:
    1. Set the affinity to the cores of the threads spawned. (Core 0 spawns the first thread and the main thread)
    2. Spawn the thread pool.
    3. Fill the map of thread ids.
*/
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
