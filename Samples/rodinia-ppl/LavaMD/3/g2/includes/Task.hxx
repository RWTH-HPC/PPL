#ifndef TASK_H
#define TASK_H

#include <atomic>
#include <memory>
#include <vector>
#include <functional>

template <typename T>
class TaskQueue;

class Task {

private:
    //The task function to be invoked
    std::function<void()> taskFunction;

protected:

public:
    Task(std::function<void()> &task_function);

    void runTask();

};

#endif //TASK_H
