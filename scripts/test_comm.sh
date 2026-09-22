#! /bin/bash
#SBATCH -N 1
#SBATCH -p qpu
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH --mem=1G

# Test comm inside qmio

IP=$(hostname -I | awk '{print $1}')

# nc -vz 10.255.3.70 5556

mkdir -p "$STORE/.qpu"
echo "$ZMQ_SERVER" > "$STORE/.qpu/zmq_server.txt"

# Start environment
source "$STORE/intel/oneapi/setvars.sh" # Start SYCL interface

# For CUNQA loading in qmio
module load qmio/hpc gcc/12.3.0 cunqa/2.4.0-python-3.11.9-mpi

./API/build/bell_state_qpu

echo "" > "$STORE/.qpu/zmq_server.txt"