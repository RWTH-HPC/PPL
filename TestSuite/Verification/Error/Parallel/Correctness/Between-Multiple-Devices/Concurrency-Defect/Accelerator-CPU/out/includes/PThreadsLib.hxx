/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/

#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>

#ifndef PTHREADLIB_HXX
#define PTHREADLIB_HXX

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
class Thread {
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
    Thread();

};

void setPool(std::vector<Thread> * pool);
std::vector<Thread> * getPool();

/**
* Definition of the Bit_Mask class for organizing which threads have to wait at a specific barrier.
* The implementation contains a pthread_barrier_t object for realizing the actual barrier.
* The parameter bitmask specifies the threads affected by the barrier. A value of 0 at position i implies, that the i'th thread has to wait at this barrier.
* The parameter limit is the number of threads affected by this barrier. (A parameter requested by the pthreads barrier implementation).*
*/
class Bit_Mask {

private:
    pthread_barrier_t barrier;
    int limit;
    std::vector<bool> mask;


public:

    // constructor for the Bit_Mask class. The size parameter depends on the size of the thread pool.
    Bit_Mask(int size, bool initial = false);

    // increases the limit by 1, used if the main thread is also affected by the barrier.
    void isMain();

    // decreases the limit by 1, used after finishing the barrier if the main thread was affected by it.
    void isNotMain();

    // the returns the number of threads affected by the barrier.
    int getLimit();

    // instatiates the barrier with the current value of limit.
    void activate();

    // returns a pointer to the pthreads barrier object to access the barrier.
    pthread_barrier_t *getBarrier();

    // sets the value of the bitmask such that the thread with the id pos will be affected by the barrier.
    void setBarrier(unsigned int pos);

    // sets the value of the bitmask such that the thread with the id pos will NOT be affected by the barrier.
    void unsetBarrier(unsigned int pos);

    // returns the value of the bit mask at position pos.
    bool get(unsigned int pos);

    // returns the number of bits encoding the bitmask.
    int size();

};

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
    Implementation of the thread specific barrier.
*/
void thread_Barrier(boost::shared_ptr<Bit_Mask> mask);

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
