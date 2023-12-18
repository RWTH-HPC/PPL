/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/

#include <atomic>
#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>
#include "TaskQueue.hxx"
#include "Task.hxx"
#include "BitMask.hxx"

#ifndef PTHREADLIB_HXX
#define PTHREADLIB_HXX

#define BUFFER_SIZE 16384

#define handle_error_en(en, msg) \
       do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)


class Thread {
    private:
    int result{};
    std::atomic<bool> concluding{};
    std::atomic<bool> blocking{};

    public:
    TaskQueue<Task> taskQueue;

    // sets the error value for creating a thread in pthreads. (might be extended for handling errors)
    void setResult(int res);

    // returns the error value for creating a thread in pthreads.
    [[nodiscard]] int getResult() const;

    // set the value of poolConcluding, necessary before the thread can join the main program flow again. (might be simplified -> only setting poolConcluding to true).
    void setConcluding(bool con);

    // returns the current value of poolConcluding.
    [[nodiscard]] bool getConcluding() const;

    //add work to the current taskqueue
    void addWork(std::function<void()>);

    // constructor to initialize all necessary variables.
    explicit Thread();

};

void addToTaskCount(uint64_t count);
void waitAllTasksDone();
void setPool(std::vector<Thread*> * pool, std::vector<pthread_t> *pThreads);
void resetThreadPool();
std::vector<Thread*> * getPool();
void setPoolSlot(Thread* thread, uint32_t index);

std::atomic<uint64_t>* getNumTasks();


/*
    Implementation of the executor which executes the jobs from the work queue until the queue is empty and the thread should return.
*/
void *execute(void *ptr);

/**
    Initialization of the pthreads setup:
    1. Set the affinity to the cores of the threads spawned. (Core 0 spawns the first thread and the main thread)
    2. Spawn the thread pool.
    3. Fill the map of thread ids.
*/
void startExecution();

/**
    Finializing function of the thread-pool and clean up.
*/
void finishExecution();

/**
    Implementation of the barrier, global barrier for all threads excluding the main thread.
    The bitmask mask defines which threads are affected by the barrier.
*/
void barrier(boost::shared_ptr<Bit_Mask> mask);

/**
    Implementation of the barrier, global barrier for all threads including the main thread.
    The bitmask mask defines which threads are affected by the barrier.
*/
void self_barrier(boost::shared_ptr<Bit_Mask> mask);


#endif
