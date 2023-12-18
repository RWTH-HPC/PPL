# Size Mapping
| Benchmark | Dataset | Splitsize | Parallel | Runtime Original | Runtime PPL 2 C18g | Runtime PPL 1 c18g | Runtime PPL 1 c18m | Runtime PPL 2 c18m | TODO | Done |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| backprop |  facenet with 65536 input features (16 hidden 1 out size are fixed) | 8192 | true | 0,02s | 0.3 | 0.13 | 0.06 | redo | Thread to GPU assignment heuristic | |
| bfs |  graph1Mnodes_6.txt (10e7 nodes with at most 18 edges)| 65536 |false | | ---- | ---- | ---- | ---- | Dynamic approach (unrealistic)| |
| cfd |  missile.domn.0.2M (about 200k nodes)| 32768 | false | | ---- | ---- | ---- | ---- | To many iterations | |
| heartwall |  video (104 Frames, 656*744 Images)| 24 |true | ~12s | 19 | 19 | 13 | 13 | Reset private array access | |
| hotspot |  power/temp (1024 * 1024 elements)| 72 |true | 0,006s | 0,06s | 0,035s | 0,032s | 0,031s | avoid redundant copies | |
| hotspot3D |  power/temp (512 * 512 * 8 elements)| 24 |true | 0.019s | 0,12s | 0,12s | 0,12s | 0,11s | | |
| Kmeans |  kdd_cup.txt (494020 * 35 features)| 65536 |true | 3.9s | 1,2s | 1,2s | 0,51s | 0,49s | Parallel centroid assignment | |
| LavaMD |  (100 * 100 * 100 * 100 elements) | 24 |true | ~100s | 39s | ~1113s | ~237s | 774s | memory locallity and small outer loop | |
| leukocyte |  (no reference data/video available)| 8192 |---- |---- |---- |---- |---- |---- | video found but split kernel is bad (interim IO because of single output) | |
| lud |  (2048 * 2048 matrix 2048.dat)| 24 |true | 0,2s | 53s | 44s | 35s | 34s | | |
| myocyte |  (workload 10, xmax 300, params.txt, y.txt)| 24 |true | 0,07s | 0,04s | 0,04s | 0,008s | 0,007s | | fixed segmentation fault (uninitialized parameter offsets) |
| nn |  4 different nn computations in sequence (42764 hurricanes each)| 1024 |true | 0,63s | 0,05s | 0,07s | 0,03s | 0,07 | | fixed segmentation fault (oob access to time0) |
| nw |  4096 * 4096 matrix| 96 | false | 0,2s |---- |---- |---- |---- | To many iterations | |
| particleFilter |  1024 * 1024 * 10 with 10000 particles| 1024 |true | 1,4s | | | | | segfault | |
| pathfinder |  100000 * 100000 area for path| 8192 | false | 7,4s |---- |---- |---- |---- | To many iterations | rework code with vectors to avoid heap-buffer overflows for large data sets |
| srad |  512 * 512| 120 |true | 0,22s | 0,45s | 0,5s | 0,14s | 0,15s | |
| streamcluster |  65536 point with dimensionality of 256| 8192 | false | 6,15s |---- |---- |---- |---- | | |




