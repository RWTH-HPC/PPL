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
#include "Task.hxx"
#include "BitMask.hxx"


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


/**
    Implementation of the thread specific barrier.
*/
void thread_Barrier(boost::shared_ptr<Bit_Mask> mask) {
    pthread_barrier_wait (mask->getBarrier());
}

