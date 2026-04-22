# MaximalBiPlex

This directory contains the baseline implementation for maximal biplex enumeration with variants such as `BPPivot` and `BPBnB`.

## Build

```bash
cmake . && make
```

## Run

```bash
bin/main -d <dataset> -q <size> -k <k> [options]
```

### Usage

```text
usage: bin/main -d=string -q=int -k=int [options] ...
options:
  -d, --data                      dataset path (string)
  -q, --lb                        size lowerbound (int)
  -k, --key                       value of k (int)
  -n, --num                       number of result (unsigned long long [=18446744073709551615])
      --no-pivoting               disable pivoting (use BPBnB instead of BPPivot)
      --no-upperbound             disable upperbound pruning
      --no-core-reduction         disable core reduction
      --no-butterfly-reduction    disable butterfly reduction
      --no-ordering               disable ordering
  -h, --help                      print this message
```

### Examples

```bash
bin/main -d datas/opsahl_ucforum.txt -q 3 -k 1
bin/main -d datas/opsahl_ucforum.txt -q 3 -k 1 --no-pivoting
bin/main -d datas/opsahl_ucforum.txt -q 3 -k 1 --no-pivoting --no-upperbound
bin/main -d datas/opsahl_ucforum.txt -q 3 -k 1 --no-butterfly-reduction
bin/main -d datas/opsahl_ucforum.txt -q 3 -k 1 --no-ordering
```
