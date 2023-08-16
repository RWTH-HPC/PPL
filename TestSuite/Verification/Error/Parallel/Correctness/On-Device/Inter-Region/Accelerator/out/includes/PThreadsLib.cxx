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

std::vector<Thread> *threadPool;

void setPool(std::vector<Thread> * pool) {
    threadPool = pool;
}

std::vector<Thread> * getPool() {
    return threadPool;
}


    // returns the pointer to the pthreads thread object which is necessary for initializing and finalizing a thread.
    pthread_t * Thread::getThread() { return &thread;}

    // returns the pthreads thread object( currently not used, might be removed in the future).
    pthread_t Thread::getThreadReal() { return thread;}

    // returns the pointer to the mutex for locking the queue (currently not used outside of the data structure, might be removed in the future).
    pthread_mutex_t *Thread::getQueueLock() {return &queueLock;}

    // sets the error value for creating a thread in pthreads. (might be extended for handling errors)
    void Thread::setResult(int res) { result = res;}

    // returns the error value for creating a thread in pthreads.
    int Thread::getResult() {return result;}

    // set the value of concluding, necessary before the thread can join the main program flow again. (might be simplified -> only setting concluding to true).
    void Thread::setConcluding(bool con) {concluding = con;}

    // returns the current value of concluding.
    bool Thread::getConcluding() {return concluding;}

    // adds a task to the ring buffer, if the buffer is not full.
    void Thread::addWork(std::function<void()> task) {
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
    std::function<void()> Thread::getTask() {
        return workBuffer.at(head);
    }

    // removes the current first task from the buffer.
    void Thread::removeFinishedJob() {
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
    bool Thread::isEmpty() {
        if (size == 0) {
            return true;
        }
        return false;
    }

    // constructor to initialize all necessary variables.
    Thread::Thread() {
        std::vector<std::function<void()>> init(BUFFER_SIZE);
        workBuffer = init;
        queueLock = PTHREAD_MUTEX_INITIALIZER;
        head = 0;
        tail = 0;
        size = 0;
        result = 0;
        concluding = false;
    }




    // constructor for the Bit_Mask class. The size parameter depends on the size of the thread pool.
    Bit_Mask::Bit_Mask(int size, bool initial) {
        std::vector<bool> init(size, initial);
        mask = init;
        if (initial == false) {
            limit = mask.size();
        } else {
            limit = 0;
        }

    }

    // increases the limit by 1, used if the main thread is also affected by the barrier.
    void Bit_Mask::isMain()  {
        limit++;
    }

    // decreases the limit by 1, used after finishing the barrier if the main thread was affected by it.
    void Bit_Mask::isNotMain() {
        limit--;
    }

    // the returns the number of threads affected by the barrier.
    int Bit_Mask::getLimit() {
        return limit;
    }

    // instatiates the barrier with the current value of limit.
    void Bit_Mask::activate() {
        pthread_barrier_init (&barrier, NULL, limit);
    }

    // returns a pointer to the pthreads barrier object to access the barrier.
    pthread_barrier_t * Bit_Mask::getBarrier() {
        return &barrier;
    }

    // sets the value of the bitmask such that the thread with the id pos will be affected by the barrier.
    void Bit_Mask::setBarrier(unsigned int pos) {
        if (mask.at(pos) != false) {
            limit++;
            mask.at(pos) = false;
        }
    }

    // sets the value of the bitmask such that the thread with the id pos will NOT be affected by the barrier.
    void Bit_Mask::unsetBarrier(unsigned int pos) {
        if (mask.at(pos) != true) {
            limit--;
            mask.at(pos) = true;
        }
    }

    // returns the value of the bit mask at position pos.
    bool Bit_Mask::get(unsigned int pos) {
        return mask.at(pos);
    }

    // returns the number of bits encoding the bitmask.
    int Bit_Mask::size() {
        return mask.size();
    }

/*
    Implementation of the executor which executes the jobs from the work queue until the queue is empty and the thread should return.
*/
void *execute(void *ptr) {
Thread *worker;
worker = (Thread *) ptr;
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
void startExecution() {
    int s, j;
    cpu_set_t cpuset;// = CPU_ALLOC(N);
    pthread_t thread;

    thread = pthread_self();

    /* Set affinity mask to include CPUs 0 to N */

    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
        handle_error_en(s, "pthread_setaffinity_np");

    /* Check the actual affinity mask assigned to the thread */

    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
        handle_error_en(s, "pthread_getaffinity_np");

    /**printf("Set returned by pthread_getaffinity_np() contained:\n");
    for (j = 0; j < CPU_SETSIZE; j++)
        if (CPU_ISSET(j, cpuset))
            printf("    CPU %d\n", j);
    */


    for(unsigned int cid = 0; cid < threadPool->size() ;cid++) {
        CPU_ZERO(&cpuset);
        CPU_SET(cid, &cpuset);


        threadPool->at(cid).setResult(pthread_create( threadPool->at(cid).getThread(), NULL, execute, (void *) &threadPool->at(cid)));

        s = pthread_setaffinity_np(threadPool->at(cid).getThreadReal(), sizeof(cpu_set_t), &cpuset);
        if (s != 0)
            handle_error_en(s, "pthread_setaffinity_np");

        s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        if (s != 0)
            handle_error_en(s, "pthread_getaffinity_np");


        if (pthread_mutex_init(threadPool->at(cid).getQueueLock(), NULL) != 0) {
            std::cout << "Mutex failed to initialize in thread:" << cid << std::endl;
        }
        if (threadPool->at(cid).getResult() != 0) {
            std::cout << "Failed to initialize thread:" << cid << std::endl;
        }
    }

    //CPU_FREE(cpuset);
}

/**
    Finializing function of the thread-pool and clean up.
*/
void finishExecution() {
    //std::cout << "Finalize" << std::endl;
    for(int i = 0; i < threadPool->size() ;i++) {
        threadPool->at(i).setConcluding(true);
    }
    for(int i = 0; i < threadPool->size() ;i++) {
        pthread_join( *threadPool->at(i).getThread(), NULL);
        //std::cout << "Thread" << i << " joined" << std::endl;
    }
}


/**
    Implementation of the thread specific barrier.
*/
void thread_Barrier(boost::shared_ptr<Bit_Mask> mask) {
    pthread_barrier_wait (mask->getBarrier());
}

/**
    Implementation of the barrier, global barrier for all threads excluding the main thread.
    The bitmask mask defines which threads are affected by the barrier.
*/
void barrier(boost::shared_ptr<Bit_Mask> mask) {
    //std::cout << "Barrier" << std::endl;
    mask->activate();
    for(int i = 0; i < mask->size(); i++) {
        if(mask->get(i) == false) {
            std::function<void()> f = [mask]() { thread_Barrier(mask); };
            //std::cout << i << std::endl;
            threadPool->at(i).addWork(f);
        }
    }
}

/**
    Implementation of the barrier, global barrier for all threads including the main thread.
    The bitmask mask defines which threads are affected by the barrier.
*/
void self_barrier(boost::shared_ptr<Bit_Mask> mask) {
    //std::cout << "Barrier" << std::endl;
    mask->isMain();
    mask->activate();
    for(int i = 0; i < mask->size(); i++) {
        if(mask->get(i) == false) {
            std::function<void()> f = [mask]() { thread_Barrier(mask); };
            //std::cout << i << std::endl;
            threadPool->at(i).addWork(f);
        }
    }
    pthread_barrier_wait (mask->getBarrier());
    mask->isNotMain();
}

