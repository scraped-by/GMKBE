# baselines

This directory contains the two baseline implementations used in the experiments reported in the paper, while preserving their original source layouts as much as possible.

- `SGMD22-MBPE`: corresponds to the paper `Efficient Algorithms for Maximal k-Biplex Enumeration` and is referred to as `iTraversal` in the reported experiments.
- `MaximalBiPlex`: corresponds to the paper `Efficient Maximal Biplex Enumerations with Improved Worst-Case Time Guarantee` and is referred to as `BPPivot` in the reported experiments.

To simplify the experimental workflow, some project files and data-loading interfaces in these baseline directories were adapted so that they can run under the build process and prepared input format used in this artifact.

## Build and Run Instructions

Both baselines use CMake and require C++17. Make sure `nlohmann_json` is available (and `fmt` for SGMD22-MBPE).

### SGMD22-MBPE (iTraversal)

```bash
cd baselines/SGMD22-MBPE
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run:

```bash
./build/main -f <prepared-file>.graph.bin -k <k> -r <num_results>
```

Arguments:
- `-f, --file` (required): path to the prepared binary graph file (`.graph.bin`)
- `-k, --k`: parameter k (default: 1)
- `-r, --r`: maximum number of results (default: 1000)
- `-q, --q`: quiet mode, suppress per-result output (default: 0)
- `-h, --help`: display help

Example:

```bash
./build/main -f ../../data/youtube.graph.bin -k 3 -r 1000000
```

### MaximalBiPlex (BPPivot)

```bash
cd baselines/MaximalBiPlex
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run:

```bash
./build/bin/main -d <dataset> -q <size_lowerbound> -k <k> -n <num_results>
```

Arguments:
- `-d, --data` (required): path to the dataset file
- `-q, --lb` (required): size lower bound (integer)
- `-k, --key` (required): value of k
- `-n, --num`: maximum number of results (default: unlimited)
- `-o, --output`: output biplex vertex sets (1 = yes, 0 = no; default: 0)
- `--no-pivoting`: disable pivoting (use BPBnB instead of BPPivot)
- `--no-upperbound`: disable upperbound pruning
- `--no-core-reduction`: disable core reduction
- `--no-butterfly-reduction`: disable butterfly reduction
- `--no-ordering`: disable ordering

Example:

```bash
./build/bin/main -d ../../data/youtube.graph.bin -q 1 -k 3 -n 1000000
```
