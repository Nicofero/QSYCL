#! /bin/bash
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p ilk
#SBATCH -t 06:00:00
#SBATCH --mem-per-cpu=3G

source $STORE/intel/oneapi/setvars.sh # Start SYCL interface

# For CUNQA loading in qmio
module load qmio/hpc gcc/12.3.0 cunqa/2.4.0-python-3.11.9-mpi

# CPU benchmarking

./build/bench_cunqa cunqa cpu 4 20 20 2

echo "CUNQA Benchmarking done"
echo "----------------------------------------------------------"
./build/bench_cunqa cpu cpu 4 20 20 4
