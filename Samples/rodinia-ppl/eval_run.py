import math
import os
import numpy as np
import pandas as pd
import fnmatch

dirs = ['Back-Propagation', 'Breadth-First-Search', 'CFD-Solver', 'Heartwall', 'Hotspot', 'Hotspot3D', 'Kmeans', 'LavaMD',
         'lud', 'myocyte', 'nn', 'nw', 'particle', 'pathfinder', 'srad', 'stream']

repetitions = 5

base_systems = ['c18m', 'c18g']
system_count = 2
iterations = 40

original = pd.read_csv("original.csv")
Indicator = 'Time:'

results = dict()
def clean(df):
    filtered = fnmatch.filter(df.columns, 'Unnamed: ??')
    df = df.drop(filtered, axis=1)
    filtered = fnmatch.filter(df.columns, 'Unnamed: ?')
    df = df.drop(filtered, axis=1)
    return df


original = clean(original)
benchmarks = original.columns


for i in range(len(dirs)):
    dir = dirs[i]
    bench = benchmarks[i]
    results[bench] = dict()
    results[bench]["original"] = round(original[bench].mean(), ndigits=4)
    for system in base_systems:
        for count in range(system_count):
            avg = math.inf
            for rep in range(repetitions):
                sum = 0
                iter = 0
                try:
                    with open(dir + '/' + system + '_' + str(count + 1) + '_' + str(rep + 1) + '.txt') as f:
                        lines = [line for line in f]
                except:
                    lines = []
                for line in lines:
                    chunked = line.split(' ')
                    if chunked[0] == Indicator and iter <= iterations:
                        iter += 1
                        sum += float(chunked[1])/1000000
                loc_avg = round(sum/iterations, 4)
                if avg > loc_avg:
                    avg = loc_avg
            results[bench][system + str(count + 1)] = avg

print(results)

print('Benchmarks\t&\tOriginal\t&\tCPU\t&\t2 CPU\t&\tGPU\t&\t2 GPU')
for dir in benchmarks:
    line = dir
    for v in results[dir]:
        line += '\t&\t' + str(results[dir][v])
    print(line)
