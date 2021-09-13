# PPL

The Parallel Pattern Language (PPL) toolchain enables the efficient development of parallel programs via parallel patterns.
The toolchain contains a high-level pattern language to develop codes, an optimization framework to optimize and schedule parallel programs globally, and a code generator to translate the optimized code to C++.

## Introduction

The toolchain processes a code given in the PPL and the Hardware Language (HL) and transforms it into the Abstract Pattern Tree (APT) and Cluster Model, respectively.
The APT is enriched with scheduling information to form the Abstract Mapping Tree (AMT).
The APT and AMT form the Parallel Pattern Intermediate Representation (PPIR).
The APT and Cluster Model information is used within the global optimization framework to determine the best optimization for the given hardware statically.
The AMT stores the optimization result by extending the APT, providing an interface for code generation.
The code generator translates the PPIR to C++ and is currently a work in progress.

A general overview of the structure of the compiler can be seen in the graphic below.

![build-1](https://user-images.githubusercontent.com/36474667/132230761-c5e8418e-1027-4989-9031-a2e49b4fc606.png)

## [Language Wiki](https://github.com/RWTH-HPC/PPL/wiki/home)

The wiki contains detailed documentation of all components of the PPL.
If you are interested and want to try out the PPL, please have a look at our [Get-Started](https://github.com/RWTH-HPC/PPL/wiki/Get-Started) page.

## Project Structure
The repository is structured as follows:

### PPL-Sources
Contains the source files of the PPL toolchain. The complete project is structured based on the seven subpackages and should be managed with an IDE supporting Maven.

### Samples
A set of various test cases written in the PPL including a translated version of the Rodinia OpenMP benchmarks ([original Rodinia](https://rodinia.cs.virginia.edu/doku.php)).

### LLVM_estimator.cpp
A small tool which parses bite-code into the LLVM IR and measures the size of the LLVM IR within the RAM. This tool can be used to compare the LLVM IR with our PPIR.

### PPL.jar

The PPL.jar is a precompiled version of the PPL compiler running on Java. The following command can be used for execution:

<code> java --add-opens java.base/java.lang=ALL-UNNAMED -jar .\PPL.jar </code>

Please have a look at the corresponding [wiki page](https://github.com/RWTH-HPC/PPL/wiki/Using-the-JAR-Tool) for further details.

### eval_LLVM.py 
A python script used to evaluate the size of the C/C++ code of the original Rodinia OpenMP benchmarks and their LLVM IR representation using the LLVM_estimator.cpp tool.

## Developing and Contributing to PPL

We greatly value any feedback and contribution to the PPL.

### Submit a Bug Report

* Bugs are tracked through [Github Issues](https://github.com/RWTH-HPC/PPL/issues).
* Please describe the bug as detailed as possible, including steps that reproduce the problem.

### Request a Feature

* Feature requests are tracked through [Github Issues](https://github.com/RWTH-HPC/PPL/issues).
* Please describe the feature as detailed as possible, including the changes you like to see compared to the current behavior.

### Make a Pull Request

* Open a new [GitHub pull request](https://github.com/RWTH-HPC/PPL/pulls) with the patch.
* Please use detailed descriptions of the purpose of the patch.
* Include the respective issue number if present.

For more information on how to develop and contribute to PPL, please contact [Adrian Schmitz](mailto:a.schmitz@itc.rwth-aachen.de).

## Related Publications

1. J. Miller, L. Trümper, C. Terboven, & Müller, M. S. “Poster: Efficiency of Algorithmic Structures”. In: IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis (SC19). Denver, Colorado, USA, 2019, pp. 1–2.

2. J. Miller, L. Trümper, C. Terboven, and M. S. Müller, “A theoretical
model for global optimization of parallel algorithms,” Mathematics,
vol. 9, no. 14, p. 1685, 2021.

3. L. Trümper, J. Miller, C. Terboven, and M. S. Müller, “Automatic mapping of parallel pattern-based algorithms on heterogeneous architectures,” in Architecture of Computing Systems. Cham: Springer
International Publishing, 2021, pp. 53–67.

## Contact Us
General: [Julian Miller](mailto:miller@itc.rwth-aachen.de)  
Implementation: [Adrian Schmitz](mailto:a.schmitz@itc.rwth-aachen.de)  

IT Center  
Group: High Performance Computing  
Division: Computational Science and Engineering  
RWTH Aachen University  
Seffenter Weg 23  
D 52074 Aachen (Germany)

## License

Copyright © 2021 IT Center, RWTH Aachen University  
This project is licensed under version 3 of the GNU General Public License.
