# ActSeekN

ActSeekN is a structural-motif-based enzyme annotation pipeline. It compares query protein structures against a curated annotation database of active-site and binding-site geometries and reports the closest matches.

## Repository Layout

- `python/`: Python entry points and helper scripts. Main entry point: `python/main.py`
- `src/cpp/`: C++ sources for the pybind11 extension. Main source: `src/cpp/ActSeekLib.cpp`
- `src/cuda/`: CUDA sources used by the extension. Main kernel: `src/cuda/distance_kernel.cu`
- `src/third_party/`: bundled third-party headers used by the extension build

The following directories are used at runtime but are not expected to be stored in the repository:

- `data/`: annotation database and input CSV files
- `structures/`: downloaded AlphaFold structures
- `results/`: output CSV files

## Requirements

### System requirements

- Python 3.10+ recommended
- C++17 compiler, for example `g++` on Linux or MSVC on Windows
- NVIDIA CUDA toolkit with `nvcc`
- Eigen headers
- Boost headers

### External libraries

- Eigen `3.4.0`
- Boost `1.86.0`
- CUDA `12.2.1`

Any CUDA `12+` version should be compatible.

Official download pages:

- Eigen 3.4.0: https://gitlab.com/libeigen/eigen/-/releases/3.4.0
- Boost 1.86.0: https://www.boost.org/releases/1.86.0/
- CUDA Toolkit archive: https://developer.nvidia.com/cuda-toolkit-archive

### Python packages

- `numpy`
- `pandas`
- `h5py`
- `requests`
- `biopython`
- `pybind11`
- `setuptools`

You can install them with:

```bash
pip install numpy pandas h5py requests biopython pybind11 setuptools
```

## Build and Installation

The extension build expects the Eigen and Boost include directories, and the CUDA installation path, to be provided through environment variables:

For example, on Linux:

```bash
export EIGEN_DIR=/path/to/eigen-3.4.0
export BOOST_DIR=/path/to/boost_1_86_0
export CUDA_PATH=/usr/local/cuda-12.2
```

On Windows, set the equivalent environment variables in PowerShell:

```powershell
$env:EIGEN_DIR="C:\path\to\eigen"
$env:BOOST_DIR="C:\path\to\boost"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
```

Build the extension in place from the repository root:

```bash
python setup.py build_ext --inplace
```

This places the compiled `ActSeekLib` module in `python/`, where `python/main.py` expects it, typically as `ActSeekLib*.so` on Linux and `ActSeekLib*.pyd` on Windows.

## Data Requirements

ActSeekN expects an annotation database in `data/`. It currently prefers:

- `data/distances_withAI_and_desc_05_03_2026.h5`

If that file is not present, it falls back to:

- the newest `data/distances_with*AI*.h5`
- `data/newDatabase_v3.pickle`

The HDF5 loader validates entries and skips malformed records before they reach the C++ layer.

## Input Format

For batch mode, pass a CSV file with an `Entry` column. `Entry` values may be empty if `Sequence` is available.

Supported columns:

- `Entry`: UniProt accession
- `Entry_seq_ref`: optional sequence reference identifier
- `Sequence`: amino-acid sequence used for fallback inference
- `EC` or `EC number`: optional EC number used to narrow fallback resolution

## Usage

### Run on a single protein

```bash
python python/main.py -p P12345
```

### Run on a CSV file

```bash
python python/main.py -f data/proteins_yeast.csv
```

### Fallback resolution

If a UniProt accession is missing or unusable, the pipeline can infer a fallback UniProt in this order:

1. `Entry_seq_ref` lookup in UniProt
2. same-EC UniProt search plus sequence ranking
3. broad HMMER sequence search

Only fallback accessions with an available AlphaFold structure are accepted.

## Outputs

### Result files

Main result files are written to `results/`:

- direct match: `actseekn-<uniprot>-results.csv`
- fallback used: `actseekn-<original>-inferred-<resolved>-results.csv`

### Additional reports for batch runs

When running with `-f`, the pipeline also writes:

- `results/<input_stem>_failed.csv`: input rows that were skipped or failed
- `results/<input_stem>_fallbacks.csv`: rows that used fallback UniProt inference

`fallbacks.csv` contains:

- `resolved_uniprot`
- `original_uniprot`
- `seq_ref`
- `inferring_method`

### Structure cache

Downloaded AlphaFold structures are stored in `structures/`.

## Network Access

The pipeline may need internet access for:

- AlphaFold structure download
- UniProt REST queries
- EBI HMMER sequence search

Rows with direct valid UniProt accessions can often run with less network work if the AlphaFold structure is already cached locally.

## SLURM Example

Example job command from the repository root:

```bash
python python/main.py -f data/proteins_yeast.csv
```

If you prefer to launch from `python/`, use:

```bash
cd python
python main.py -f ../data/proteins_yeast.csv
```

## Notes

- The process pool size is based on `SLURM_CPUS_PER_TASK` when available, otherwise `os.cpu_count() // 4`.
- Existing result files are reused instead of recomputed.
- The pipeline creates `results/` and `structures/` automatically if they do not already exist.
