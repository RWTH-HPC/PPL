/**


This file implements the Thread Pool and barriers.


*/
#define _GNU_SOURCE 1
#define _GNU_SOURCE 1
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <array>
#include <atomic>
#include <chrono>
#include <latch>
#include <unordered_map>
#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>
#include "PThreadsLib.hxx"
#include "Task.hxx"
#include "BitMask.hxx"


std::vector<pthread_t> *threadPoolPThreadIds;
std::vector<Thread*> *threadPool;

std::atomic<uint64_t> numTasks = 0;
std::atomic<bool> poolConcluding = false;
std::latch *latch;

void addToTaskCount(uint64_t count) {
    numTasks.fetch_add(count, std::memory_order::relaxed);
}

void waitAllTasksDone() {
    auto tasksToRun = numTasks.load(std::memory_order::relaxed);
    while (tasksToRun > 0) {
        numTasks.wait(tasksToRun, std::memory_order::relaxed);
        tasksToRun = numTasks.load(std::memory_order::relaxed);
    }
}

void setPool(std::vector<Thread*> * pool, std::vector<pthread_t> *pThreads) {
    threadPoolPThreadIds = pThreads;
    threadPool = pool;
}

void resetThreadPool() {
    std::fill(threadPoolPThreadIds->begin(), threadPoolPThreadIds->end(),0);
    std::fill(threadPool->begin(), threadPool->end(),nullptr);
}

std::vector<Thread*> * getPool() {
    return threadPool;
}

void setPoolSlot(Thread* thread, uint32_t index) {
    if (threadPool == nullptr) {
        std::cout << "Thread pool initialization not possible! Pool is nullptr! Index: " << index << std::endl;
        return;
    }
    if (threadPool->size() <= index) {
        std::cout << "Thread pool initialization not possible! Size: " << threadPool->size() << " Index: " << index << std::endl;
        return;
    }

    if (threadPool->at(index) == nullptr) {
        threadPool->at(index) = thread;
    } else {
        std::cout << "Thread pool already initialized at index! Size: " << threadPool->size() << " Index: " << index << std::endl;
        return;
    }
}

//Get ordering guarantees:
//After returning from this function, the thread pool init is finished
void waitForThreadPoolInitialization(std::latch* latch) {
    latch->arrive_and_wait();
}

std::atomic<uint64_t>* getNumTasks() {
    return &numTasks;
}

    // sets the error value for creating a thread in pthreads. (might be extended for handling errors)
    void Thread::setResult(int res) { result = res;}

    // returns the error value for creating a thread in pthreads.
    int Thread::getResult() const {return result;}

    // set the value of concluding, necessary before the thread can join the main program flow again. (might be simplified -> only setting concluding to true).
    void Thread::setConcluding(bool con) {
        concluding.store(con, std::memory_order_relaxed);
        if (this->blocking.load(std::memory_order::relaxed)) {
            //Todo find a better way to unblock atomic wait
            this->taskQueue.tail.fetch_add(BUFFER_SIZE, std::memory_order_relaxed);
            this->taskQueue.tail.notify_all();
        }
    }

    // returns the current value of concluding.
    bool Thread::getConcluding() const {return concluding.load(std::memory_order_relaxed);}

    // constructor to initialize all necessary variables.
    Thread::Thread() {
        this->taskQueue.init();
        this->result = 0;
        this->concluding = false;
        this->blocking = false;
    }

void concludeAllThreads();

struct execute_startup_data {
    uint32_t threadIndex;
    std::latch* latch;
};

[[nodiscard]] Task* pollTask(TaskQueue<Task> *queue) {
    Task* task = queue->pollTask(nullptr);
    if (task != nullptr)
        return task;

    return nullptr;
}
    // adds a task to the ring buffer, if the buffer is not full.
    void Thread::addWork(std::function<void()> task) {
        Task* taskwrapper = new Task(task);
        bool res = taskQueue.offerTask(taskwrapper);
    }

/*
    Implementation of the executor which executes the jobs from the work queue until the queue is empty and the thread should return.
*/
void *execute(void* ptr) {
    auto* startupData = (struct execute_startup_data*)ptr;
    auto threadIndex = startupData->threadIndex;
    auto latch = startupData->latch;
    Thread worker = Thread();
    setPoolSlot(&worker, threadIndex);
    waitForThreadPoolInitialization(latch);
    delete startupData;

    uint64_t totalTasksPolled = 0;

    auto workerPtr = &worker;

    Task* task = pollTask(&worker.taskQueue);


    while(true) {
        while (task != nullptr) {
            totalTasksPolled++;
            task->runTask();
            delete task;
            task = pollTask(&worker.taskQueue);
        }
        if (totalTasksPolled > 0) {
            uint64_t totalAvailable = numTasks.fetch_sub(totalTasksPolled, std::memory_order_relaxed);
            numTasks.notify_all();
            totalAvailable -= totalTasksPolled;
            totalTasksPolled = 0;

            if (totalAvailable == 0 && poolConcluding.load(std::memory_order_relaxed)) {
                concludeAllThreads();
            }
        }
        if (!worker.getConcluding()) {
            task = pollTask(&worker.taskQueue);
            if (worker.getConcluding()) {
                break;
            }
        } else {
            break;
        }
    }
    std::cout.flush();
    pthread_exit(nullptr);
}



/**
    Initialization of the pthreads setup:
    1. Set the affinity to the cores of the threads spawned. (Core 0 spawns the first thread and the main thread)
    2. Spawn the thread pool.
    3. Fill the map of thread ids.
*/
void startExecution() {
    resetThreadPool();

    int s;
    cpu_set_t cpuset;// = CPU_ALLOC(N);
    pthread_t thread;

    thread = pthread_self();
    /* Set affinity mask to include CPUs 0 to N */

    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
        handle_error_en(s, "pthread_setaffinity_np");

//    /* Check the actual affinity mask assigned to the thread */
//    //This is actually not checking anything useful! This doesn't actually check the affinity, it just stores it in cpuset!
//    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
//    if (s != 0)
//        handle_error_en(s, "pthread_getaffinity_np");

    /**printf("Set returned by pthread_getaffinity_np() contained:\n");
    for (j = 0; j < CPU_SETSIZE; j++)
        if (CPU_ISSET(j, cpuset))
            printf("    CPU %d\n", j);
    */
    uint32_t size = threadPool->size();

    latch = new std::latch(size+1);

    for (int j = 0; j < size; j++) {

            CPU_ZERO(&cpuset);
            CPU_SET(j+1, &cpuset); //Keep Core 0 exclusive for main thread


            pthread_t pthread_id;
            auto startup_data = new struct execute_startup_data;
            startup_data->threadIndex = j;
            startup_data->latch = latch;

            //Set thread affinity at pthread creation, so the thread cannot initialize on the wrong cpu before setting affinity
            pthread_attr_t pthread_attr;
            s = pthread_attr_init(&pthread_attr);
            if (s != 0) {
                handle_error_en(s, "pthread_attr_init");
            }
            s = pthread_attr_setaffinity_np(&pthread_attr, sizeof(cpu_set_t), &cpuset);
            if (s != 0) {
                handle_error_en(s, "pthread_attr_setaffinity_np");
            }
            s = pthread_create(&pthread_id, &pthread_attr, execute, (void*)startup_data);
            pthread_attr_destroy(&pthread_attr);
            if (s != 0) {
                std::cout << "Failed to initialize thread:" << j << std::endl;
            }

            threadPoolPThreadIds->push_back(pthread_id);

    }
    waitForThreadPoolInitialization(latch);

    //CPU_FREE(cpuset);
}

/**
    Finalizing function of the thread-pool and clean up.
*/
void finishExecution() {
    poolConcluding.store(true, std::memory_order_relaxed);
    concludeAllThreads();

    for(unsigned int i = 0; i < threadPool->size() ;i++) {
        void* exitStatusPtr = nullptr;
        pthread_join(threadPoolPThreadIds->at(i), &exitStatusPtr);
        if (exitStatusPtr != nullptr) {
            std::cout << "Thread" << i << " exit status is unexpected:" << exitStatusPtr << std::endl;
        }
    }
    delete latch;
}

void concludeAllThreads() {
    for(Thread *i : *threadPool) {
        i->setConcluding(true);
    }
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
            threadPool->at(i)->addWork(f);
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
            threadPool->at(i)->addWork(f);
        }
    }
    pthread_barrier_wait (mask->getBarrier());
    mask->isNotMain();
}

