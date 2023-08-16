Structure:

The folders in this directory define the different tests to be performed.

General:
    The general test cases cover the basic structures of the code generation, like expressions, function calls and control flow constructs.
    
Plain:
    The plain test cases cover the the five basic parallel patterns provided in the front-end (Map, Reduction, Stencil, Dynamic Programming and Recursion).
    All patterns are executed in a simple example on different target specifications. The specifications are constructed from a combination of CPU, GPU and Inter Node communication.
    For these test cases to execute correctly a setup composed of two nodes with two CPU sockets and two accelerators (GPUs) is necessary.
    
Nested:
    The nested test cases cover a parallel pattern within another map pattern. These test cases aim to cover the sequential generation of parallel patterns.
    For these tests to execute correctly a setup composed of a node with two sockets and an accelerator (GPU) is necessary.
    
Concatenated:
    The concatenated test cases cover the generation of multiple patterns executed in succession in different scenarios, e.g., synchronization between multiple nodes etc..
    For these test cases to execute correctly a setup composed of two nodes with two CPU sockets and two accelerators (GPUs) is necessary.