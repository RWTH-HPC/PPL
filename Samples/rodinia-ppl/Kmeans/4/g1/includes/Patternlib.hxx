
/*********************************************************************/
/*        This is a generated C++ file.                              */
/*        Generated by PatternDSL.                                   */
/*        Contains the standard library header                       */
/*********************************************************************/
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#ifndef PATTERNLIB_HXX
#define PATTERNLIB_HXX

#define noop



inline long long get_time() {
    auto start = std::chrono::high_resolution_clock::now();
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count();
    return microseconds;
}

inline long long get_time_nano() {
    auto start = std::chrono::high_resolution_clock::now();
    long long nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    return nanoseconds;
}

std::vector<std::string> split(std::string base) ;

template<typename T>
std::vector<double> d_read(T file, long long size) {
	std::vector<double> v(size);
	std::string line;
	std::ifstream fin(file);
	long long i = 0;
	if (fin.is_open()) {
	    while(std::getline(fin,line)) {
	        for(auto x : split(line)) {
            	std::istringstream iss(x);
            	double in;
            	if(!(iss >> in)) {break;}
            	v[i] = in;
            	i++;
            }
	    }
	    fin.close();
	} else {
	    std::cout << "Unable to open file" << std::endl;
	}
	if (i != size) {
	    std::cout << "Array size mismatch" << std::endl;
	    std::cout << "Expected:" << size << std::endl;
        std::cout << "Provided:" << i << std::endl;
	}
	return v;
}

template<typename T>
std::vector<float> f_read(T file, long long size) {
	std::vector<float> v(size);
	std::string line;
	std::ifstream fin(file);
	long long i = 0;
	if (fin.is_open()) {
	while(std::getline(fin,line)) {
	    for(auto x : split(line)) {
            std::istringstream iss(x);
            float in;
            if(!(iss >> in)) {break;}
            v[i] = in;
            i++;
        }
	}
	    fin.close();
	} else {
	    std::cout << "Unable to open file" << std::endl;
	}
	if (i != size) {
	    std::cout << "Array size mismatch" << std::endl;
	    std::cout << "Expected:" << size << std::endl;
        std::cout << "Provided:" << i << std::endl;
	}
	return v;
}
template<typename T>
std::vector<int> i_read(T file, long long size) {
	std::vector<int> v(size);
	std::string line;
	std::ifstream fin(file);
	long long i = 0;
	if (fin.is_open()) {
	while(std::getline(fin,line)) {
	    for(auto x : split(line)) {
            std::istringstream iss(x);
            int in;
            if(!(iss >> in)) {break;}
            v[i] = in;
            i++;
        }
	}
	    fin.close();
	} else {
	    std::cout << "Unable to open file" << std::endl;
	}
	if (i != size) {
	    std::cout << "Array size mismatch" << std::endl;
	    std::cout << "Expected:" << size << std::endl;
        std::cout << "Provided:" << i << std::endl;
	}
	return v;
}
template<typename T>
std::vector<std::string> read(T file, long long size) {
	std::vector<std::string> v(size);
	std::string line;
	std::ifstream fin(file);
	long long i = 0;
	if (fin.is_open()) {
	while(std::getline(fin,line)) {
	    for(auto x : split(line)) {
            std::istringstream iss(x);
            std::string in;
            if(!(iss >> in)) {break;}
            v[i] = in;
            i++;
        }
	}
	    fin.close();
	} else {
	    std::cout << "Unable to open file" << std::endl;
	}
	if (i != size) {
	    std::cout << "Array size mismatch" << std::endl;
	    std::cout << "Expected:" << size << std::endl;
        std::cout << "Provided:" << i << std::endl;
	}
	return v;
}

#endif