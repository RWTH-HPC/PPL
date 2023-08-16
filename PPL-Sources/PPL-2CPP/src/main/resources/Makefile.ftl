MPICC := mpicxx
CUDACC := nvcc
CC := g++

LIB := thread_lib.o pattern_lib.o BitMask.o Task.o
CU_LIB := cuda_pool_lib.o

FLAGS := -std=c++20 -O3 -pthread -g #-fsanitize=address
CFLAGS:= -arch=sm_70 -pthread -forward-unknown-to-host-compiler -O3 -std=c++17 -g #-fsanitize=address
LFLAGS:= -pthread -O3 -std=c++20
LCFLAGS:= -L$$CUDA_ROOT/lib64 -lcudart -lm

.SUFFIXES: .exe

SRC = $(wildcard *.cxx)

OBJ = $(SRC:%.cxx=obj/%.o)

EXE = $(OBJ:obj/%.o=bin/%.exe)

all: $(EXE)

bin/%.exe: obj/%.o $(LIB) $(CU_LIB) cuda/%.o
	$(MPICC) $(FLAGS) $^ -o $@ $(LCFLAGS)

thread_lib.o: includes/PThreadsLib.cxx
	$(CC) $(LFLAGS) -c -o $@ $<

cuda/%.o: includes/cuda_lib_%.cu
	$(CUDACC) $(CFLAGS) -c -o $@ $<

pattern_lib.o: includes/Patternlib.cxx
	$(CC) $(FLAGS) -c -o $@ $<

BitMask.o: includes/BitMask.cxx
	$(CC) $(FLAGS) -c -o $@ $<

Task.o: includes/Task.cxx
	$(CC) $(FLAGS) -c -o $@ $<
	
cuda_pool_lib.o: includes/cuda_pool_lib.cxx
	$(CUDACC) $(CFLAGS) -c -o $@ $<

obj/%.o: %.cxx
	$(MPICC) $(FLAGS) -c -o $@ $<



clean:
	rm bin/* -f
	rm cuda/* -f
	rm obj/* -f
	rm *.o


run: $(EXE)
	 $$MPIEXEC -np 2 -H login18-g-1 $<

runall: $(EXE)
	for i in $(EXE) ; do echo $$i; $$MPIEXEC $$FLAGS_MPI_BATCH $$i ; done