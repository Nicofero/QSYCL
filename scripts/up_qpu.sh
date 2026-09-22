#! /bin/bash
#SBATCH -N 1
#SBATCH -p qpu
#SBATCH -n 1
#SBATCH -t 00:03:00
#SBATCH --mem=1G

IP=$(hostname -I | awk '{print $1}')

nc -vz 10.255.3.70 5556

mkdir -p "$STORE/.qpu"
echo "$ZMQ_SERVER" > "$STORE/.qpu/zmq_server.txt"

TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
TIME_LIMIT_SECONDS=$(echo "${TIME_LIMIT}" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
sleep ${TIME_LIMIT_SECONDS}s

echo "" > "$STORE/.qpu/zmq_server.txt"