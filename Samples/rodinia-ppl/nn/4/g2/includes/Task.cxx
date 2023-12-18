#include <iostream>
#include "Task.hxx"
#include "TaskQueue.hxx"

void Task::runTask() {
    taskFunction();
}

Task::Task(std::function<void()> &task_function) {
    this->taskFunction = task_function;
}
