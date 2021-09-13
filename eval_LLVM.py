import os
import subprocess

mainpath = os.getcwd()

Test_Cases = ['b+tree', 'backprop', 'bfs', 'cfd', 'heartwall', 'hotspot', 'hotspot3D', 'kmeans', 'lavaMD', 'leukocyte', 'lud', 'myocyte', 'nn', 'nw', 'particlefilter', 'pathfinder', 'srad', 'streamcluster']

original_size = []
IR_size = []

idx = 0


def update_idx(index):
    index+=1
    print(Test_Cases[index-1])
    return index


def getllvmsize(command):
    if os.path.exists("temp.txt"):
        os.remove("temp.txt")
    f = open("temp.txt", "w")
    f.close()
    os.system(command + " > temp.txt")
    with open('temp.txt') as f:
        res = f.readlines()
    f.close()
    os.remove("temp.txt")
    return int(res[0][:-6])


for _ in Test_Cases:
    original_size.append(0)
    IR_size.append(0)


def parsefiles(benchmark):
    if benchmark == 'leukocyte':
        os.chdir(mainpath + '/' + benchmark + '/OpenMP')
    else:
        os.chdir(mainpath + '/' + benchmark)
    if benchmark == 'cfd':
        original_size[idx] += os.path.getsize('euler3d_cpu.cpp')
        os.system("clang++ -Wno-everything -S -emit-llvm euler3d_cpu.cpp")
        size = getllvmsize(mainpath + '/LLVMEst.exe ' + 'euler3d_cpu.ll')
        print(size)
        IR_size[idx] += size
    elif benchmark == 'heartwall':
        for path, directories, files in os.walk('.'):
            os.chdir(path)
            for file in files:
                if file.endswith(".c") or file.endswith(".cpp"):
                    original_size[idx] += os.path.getsize(file)
        os.system("clang -Wno-everything -S -emit-llvm main.c")
        os.system("clang -Wno-everything -S -emit-llvm avilib.c")
        os.system("clang -Wno-everything -S -emit-llvm avimod.c")
        size = getllvmsize(mainpath + '/LLVMEst.exe ' + 'main.ll')
        print(size)
        IR_size[idx] += size
        size = getllvmsize(mainpath + '/LLVMEst.exe ' + 'avilib.ll')
        print(size)
        IR_size[idx] += size
        size = getllvmsize(mainpath + '/LLVMEst.exe ' + 'avimod.ll')
        print(size)
        IR_size[idx] += size
    elif benchmark == 'myocyte':
        for path, directories, files in os.walk('.'):
            os.chdir(path)
            for file in files:
                if file.endswith('.c'):
                    original_size[idx] += os.path.getsize(file)
        os.system("clang -Wno-everything -S -emit-llvm main.c")
        size = getllvmsize(mainpath + '/LLVMEst.exe main.ll')
        print(size)
        IR_size[idx] += size
    elif benchmark == 'nn':
        original_size[idx] += os.path.getsize('nn_openmp.c')
        os.system("clang -Wno-everything -S -emit-llvm nn_openmp.c")
        size = getllvmsize(mainpath + '/LLVMEst.exe ' + 'nn_openmp.ll')
        print(size)
        IR_size[idx] += size
    else:
        for path, directories, files in os.walk('.'):
            os.chdir(path)
            for file in files:
                if file.endswith(".c"):
                    original_size[idx] += os.path.getsize(file)
                    os.system("clang -Wno-everything -S -emit-llvm " + file)
                    size = getllvmsize(mainpath + '/LLVMEst.exe ' + file[:-2] + '.ll')
                    print(size)
                    IR_size[idx] += size
                if file.endswith(".cpp"):
                    original_size[idx] += os.path.getsize(file)
                    os.system("clang++ -Wno-everything -S -emit-llvm " + file)
                    size = getllvmsize(mainpath + '/LLVMEst.exe ' + file[:-4] + '.ll')
                    print(size)
                    IR_size[idx] += size


for x in Test_Cases:
    parsefiles(x)
    idx = update_idx(idx)

os.chdir(mainpath)
if os.path.exists("Names.txt"):
    os.remove("Names.txt")
f = open("Names.txt", "w")
for x in Test_Cases:
    f.write(x + "\n")

if os.path.exists("Orig_size.txt"):
    os.remove("Orig_size.txt")
f = open("Orig_size.txt", "w")
for x in original_size:
    f.write(str(x) + "\n")

if os.path.exists("IR_size.txt"):
    os.remove("IR_size.txt")
f = open("IR_size.txt", "w")
for x in IR_size:
    f.write(str(x) + "\n")
