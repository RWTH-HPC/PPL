/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/

#include <boost/config.hpp>
#include <boost/smart_ptr.hpp>

#ifndef BITMASK_HXX
#define BITMASK_HXX

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



/**
    Implementation of the thread specific barrier.
*/
void thread_Barrier(boost::shared_ptr<Bit_Mask> mask);


#endif
