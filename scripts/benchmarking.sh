#! /bin/bash
#SBATCH -J qsycl_bench
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p ilk
#SBATCH -t 06:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH -o bench_vqe_%j.out

# General benchmark. Two programs from examples/benchmarking/:
#   ./build/bench_vqe      VQE over every execution method (native / host /
#                          sequential) for each backend, plus the CUNQA vQPU sweep.
#   ./build/bench_circuit  single-circuit latency and qubit limits, no batching.
#
# Usage:
#   sbatch scripts/benchmarking.sh [cpu|gpu|cunqa|qmio|circuit|all]
#     cpu / gpu / cunqa / qmio : VQE benchmark on that backend
#     circuit                  : circuit benchmark (no VQE) on $CIRCUIT_DEVICES:
#                                latency of one circuit + throughput of B circuits
#                                (CUNQA throughput swept over $CIRCUIT_VQPUS)
#     all (default)            : cpu + cunqa + circuit
#
#   The SBATCH header above targets the CPU partition. GPU and QMIO runs need
#   their own partition, so override it on the command line, e.g.:
#     sbatch -p <gpu-partition> --gres=gpu:1 scripts/benchmarking.sh gpu
#     sbatch -p qpu -c 1 -t 01:00:00          scripts/benchmarking.sh qmio
#
#   Every knob below can be overridden without editing this file, e.g.:
#     sbatch --export=ALL,MAX_Q=16,CUNQA_VQPUS="1 2 4 8 16" scripts/benchmarking.sh cunqa
#   sbatch --export splits on commas, so pass comma lists through the environment:
#     CIRCUIT_BATCH_SIZES=1,8,32,128 sbatch --export=ALL,CIRCUIT_MODES=throughput scripts/benchmarking.sh circuit
#
# Local run (own machine/GPU, no Slurm): run it with bash instead of sbatch, e.g.
#     scripts/benchmarking.sh gpu
#     CIRCUIT_DEVICES=gpu CIRCUIT_MAX_Q=28 scripts/benchmarking.sh circuit
#   Detected automatically when SLURM_JOB_ID is unset (force with LOCAL=0/1). Locally:
#     - oneAPI comes from ONEAPI_SETVARS (default ~/intel/oneapi/setvars.sh), no `module load`;
#     - `all` runs cpu + gpu + circuit, and CIRCUIT_DEVICES defaults to cpu,gpu (no CUNQA);
#     - the circuit memory guard is 80% of the machine's RAM (CPU) or of the GPU's memory (GPU);
#     - results go to results/benchmarking/local_<date>_<time>/, with system.txt describing the machine.
#   The benchmarks must be built: cmake -B build -DENABLE_GPU_BACKEND=ON -DBUILD_BENCH=ON
#
# Results: results/benchmarking/<jobid>/{<mode>.csv,<mode>.log}
#          (circuit runs: circuit.csv, circuit_throughput.csv, circuit.log)
# Plots:   results/benchmarking/analysis.ipynb

set -uo pipefail

MODE="${1:-all}"

# Local (no Slurm) or cluster run
if [[ -z "${LOCAL:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then LOCAL=0; else LOCAL=1; fi
fi
if [[ "$LOCAL" == "1" ]]; then
    # Paths below are relative to the repository root.
    cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
fi

# ---------------------------------------------------------------- knobs ----
BIN="${BIN:-./build/bench_vqe}"
CIRCUIT_BIN="${CIRCUIT_BIN:-./build/bench_circuit}"
HAMILTONIAN="${HAMILTONIAN:-ising}"     # ising (2 measurement groups) | heisenberg (3)
REPS="${REPS:-2}"                       # RealAmplitudes depth
ITERS="${ITERS:-10}"                    # iterations per native/host run
SEQ_ITERS="${SEQ_ITERS:-2}"             # iterations per sequential run
SHOTS="${SHOTS:-1024}"                  # shots per circuit (CUNQA / QMIO)

# Exact simulators (CPU / GPU)
MIN_Q="${MIN_Q:-4}"
MAX_Q="${MAX_Q:-20}"
STEP_Q="${STEP_Q:-2}"
OPTIMIZERS_CPU="${OPTIMIZERS_CPU:-cpu}"
OPTIMIZERS_GPU="${OPTIMIZERS_GPU:-cpu,gpu}"

# CUNQA: qubit range and the vQPU counts to compare
CUNQA_MIN_Q="${CUNQA_MIN_Q:-4}"
CUNQA_MAX_Q="${CUNQA_MAX_Q:-20}"
CUNQA_STEP_Q="${CUNQA_STEP_Q:-4}"
CUNQA_VQPUS="${CUNQA_VQPUS:-1 2 4 8}"
# Raise the vQPUs from this script (1) or use ones already raised (0).
CUNQA_RAISE="${CUNQA_RAISE:-1}"
# Every vQPU configuration gets the same total cores, to compare fairly with
# QSYCL's 64-core CPU runs: k vQPUs are raised with CUNQA_NODE_CORES / k cores
# each (1 -> 64, 2 -> 32, 4 -> 16, 8 -> 8). Each vQPU count is therefore raised,
# benchmarked and dropped on its own. Only applies when CUNQA_RAISE=1.
CUNQA_NODE_CORES="${CUNQA_NODE_CORES:-64}"
QRAISE_CORES_FLAG="${QRAISE_CORES_FLAG:--c}"   # qraise option for cores per vQPU
# A vQPU that dies (e.g. OOM-killed: memory is split like the cores, so k vQPUs
# get ~1/k of the node's memory each) never answers. After this long without a
# reply it is treated as dead instead of hanging the job until its time limit.
# Keep it above the longest legitimate single-circuit time.
export CUNQA_RECV_TIMEOUT_MS="${CUNQA_RECV_TIMEOUT_MS:-900000}"   # 15 min
QRAISE_TIME="${QRAISE_TIME:-05:00:00}"
# Extra qraise flags. --co-located makes the vQPUs reachable over TCP from this
# job's node. Check them against `qraise --help` for your CUNQA version.
QRAISE_EXTRA="${QRAISE_EXTRA:---co-located}"
QRAISE_WAIT_S="${QRAISE_WAIT_S:-600}"   # max time to wait for the vQPUs to register

# QMIO: real hardware -- keep it small
QMIO_MIN_Q="${QMIO_MIN_Q:-2}"
QMIO_MAX_Q="${QMIO_MAX_Q:-4}"
QMIO_STEP_Q="${QMIO_STEP_Q:-2}"
QMIO_ITERS="${QMIO_ITERS:-3}"

# Single circuit (bench_circuit): sweep qubits until a limit is hit
if [[ "$LOCAL" == "1" ]]; then
    CIRCUIT_DEVICES="${CIRCUIT_DEVICES:-cpu,gpu}"
else
    CIRCUIT_DEVICES="${CIRCUIT_DEVICES:-cpu,cunqa}"
fi
CIRCUIT_KINDS="${CIRCUIT_KINDS:-real_amplitudes,ghz,qft}"
CIRCUIT_MIN_Q="${CIRCUIT_MIN_Q:-2}"
CIRCUIT_MAX_Q="${CIRCUIT_MAX_Q:-34}"
CIRCUIT_STEP_Q="${CIRCUIT_STEP_Q:-1}"
CIRCUIT_REPEATS="${CIRCUIT_REPEATS:-5}"
CIRCUIT_TIME_LIMIT="${CIRCUIT_TIME_LIMIT:-60}"   # seconds per circuit before the sweep stops
CIRCUIT_MODES="${CIRCUIT_MODES:-latency,throughput}"
CIRCUIT_BATCH_SIZES="${CIRCUIT_BATCH_SIZES:-1,8,32,128}"   # throughput: circuits per submission
CIRCUIT_TP_QUBITS="${CIRCUIT_TP_QUBITS:-4,8,12,16}"        # throughput: qubit counts
CIRCUIT_VQPUS="${CIRCUIT_VQPUS:-1 2 4 8}"                  # throughput on CUNQA: vQPU counts
# Exact-simulator memory guard, in GB (0 = no guard). Setting CIRCUIT_MAX_MEM_GB
# applies that value to every device; otherwise each device gets 80% of its own
# memory (see mem_guard_gb).
CIRCUIT_MAX_MEM_GB="${CIRCUIT_MAX_MEM_GB:-}"

# Default guard for one device: 80% of this job's memory (CPU, from Slurm or, locally,
# from /proc/meminfo) or of the first visible GPU's memory (GPU, from nvidia-smi).
mem_guard_gb() {
    local dev="$1"
    if [[ -n "$CIRCUIT_MAX_MEM_GB" ]]; then echo "$CIRCUIT_MAX_MEM_GB"; return; fi
    if [[ "$dev" == "gpu" ]]; then
        local mib
        mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ "$mib" =~ ^[0-9]+$ ]]; then echo $(( mib * 8 / 10 / 1024 )); else echo 0; fi
    elif [[ -n "${SLURM_MEM_PER_CPU:-}" && -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
        echo $(( SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK * 8 / 10 / 1024 ))
    elif [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
        echo $(( SLURM_MEM_PER_NODE * 8 / 10 / 1024 ))
    elif [[ "$LOCAL" == "1" ]]; then
        awk '/^MemTotal:/ { printf "%d\n", $2 * 0.8 / 1024 / 1024 }' /proc/meminfo
    else
        echo 0
    fi
}

# ---------------------------------------------------------- environment ----
if [[ "$LOCAL" == "1" ]]; then
    ONEAPI_SETVARS="${ONEAPI_SETVARS:-$HOME/intel/oneapi/setvars.sh}"
else
    ONEAPI_SETVARS="${ONEAPI_SETVARS:-${STORE:-}/intel/oneapi/setvars.sh}"
fi
if [[ -z "${SETVARS_COMPLETED:-}" ]]; then        # not already sourced in this shell
    if [[ ! -f "$ONEAPI_SETVARS" ]]; then
        echo "error: oneAPI setvars.sh not found at $ONEAPI_SETVARS (set ONEAPI_SETVARS)" >&2
        exit 1
    fi
    # setvars.sh is not written for `set -u`
    set +u; source "$ONEAPI_SETVARS" > /dev/null; set -u   # Start SYCL interface
fi

if [[ "$LOCAL" != "1" ]]; then
    # For CUNQA loading in qmio
    module load qmio/hpc gcc/12.3.0 cunqa/2.4.0-python-3.11.9-mpi
fi

for b in "$BIN" "$CIRCUIT_BIN"; do
    if [[ ! -x "$b" ]]; then
        echo "error: $b not found -- build first (cmake -B build -DBUILD_BENCH=ON ...; cmake --build build --target bench_vqe bench_circuit)" >&2
        exit 1
    fi
done

OUT_DIR="results/benchmarking/${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
echo "Results in $OUT_DIR"

# What this run ran on (the plan asks to record it next to the numbers).
{
    echo "date:    $(date -Iseconds)"
    echo "host:    $(hostname)"
    echo "mode:    $MODE (local=$LOCAL)"
    echo "commit:  $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (with local changes)')"
    echo "cpu:     $(lscpu 2>/dev/null | awk -F': *' '/^Model name/ {print $2; exit}') ($(nproc) cores available)"
    echo "memory:  $(awk '/^MemTotal:/ {printf "%.1f GB\n", $2 / 1024 / 1024}' /proc/meminfo)"
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    fi
    echo "sycl:"
    sycl-ls 2>/dev/null | sed 's/^/  /'
} > "$OUT_DIR/system.txt"

COMMON=(--hamiltonian "$HAMILTONIAN" --reps "$REPS" --iters "$ITERS"
        --seq-iters "$SEQ_ITERS" --shots "$SHOTS")

section() {
    echo
    echo "=========================================================="
    echo " $*"
    echo "=========================================================="
}

# ------------------------------------------------------------- CPU / GPU ----
run_cpu() {
    section "CPU state-vector simulator"
    "$BIN" "${COMMON[@]}" --devices cpu --optimizers "$OPTIMIZERS_CPU" \
        --methods native,host,sequential \
        --min-qubits "$MIN_Q" --max-qubits "$MAX_Q" --step "$STEP_Q" \
        --csv "$OUT_DIR/cpu.csv" 2>&1 | tee "$OUT_DIR/cpu.log"
}

run_gpu() {
    section "GPU state-vector simulator"
    "$BIN" "${COMMON[@]}" --devices gpu --optimizers "$OPTIMIZERS_GPU" \
        --methods native,host,sequential \
        --min-qubits "$MIN_Q" --max-qubits "$MAX_Q" --step "$STEP_Q" \
        --csv "$OUT_DIR/gpu.csv" 2>&1 | tee "$OUT_DIR/gpu.log"
}

# ----------------------------------------------------------------- CUNQA ----
CUNQA_REGISTRY="${STORE:-$HOME}/.cunqa/qpus.json"

registered_vqpus() {
    python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$CUNQA_REGISTRY" 2>/dev/null || echo 0
}

# Cores per vQPU so that k vQPUs together use CUNQA_NODE_CORES.
cores_per_vqpu() {
    local k="$1"
    if (( CUNQA_NODE_CORES % k != 0 )); then
        echo "warning: $CUNQA_NODE_CORES cores don't split evenly over $k vQPUs, using $(( CUNQA_NODE_CORES / k )) each" >&2
    fi
    echo $(( CUNQA_NODE_CORES / k ))
}

# Waits until the registry is empty, so the next configuration can't pick up
# vQPUs from the previous one (CUNQA_MAX_VQPUS takes the first k entries).
wait_registry_empty() {
    local waited=0
    while (( $(registered_vqpus) > 0 )); do
        if (( waited >= QRAISE_WAIT_S )); then
            echo "error: $(registered_vqpus) vQPU(s) still in $CUNQA_REGISTRY ${QRAISE_WAIT_S}s after qdrop" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}

drop_vqpus() {
    if [[ "$CUNQA_RAISE" == "1" ]]; then
        echo "Dropping the vQPUs raised by this benchmark"
        qdrop --all || true
        wait_registry_empty
    fi
}

# raise_vqpus <count>: raises <count> vQPUs with CUNQA_NODE_CORES/<count> cores each.
raise_vqpus() {
    local wanted="$1"
    if [[ "$CUNQA_RAISE" != "1" ]]; then
        echo "CUNQA_RAISE=0: using the $(registered_vqpus) vQPU(s) already in $CUNQA_REGISTRY"
        return 0
    fi
    # Start from an empty registry: leftovers would be counted and benchmarked.
    if (( $(registered_vqpus) > 0 )); then
        echo "Registry not empty ($(registered_vqpus) vQPU(s)), dropping them first"
        drop_vqpus || return 1
    fi
    local cores
    cores=$(cores_per_vqpu "$wanted")
    echo "Raising $wanted vQPU(s) x $cores cores: qraise -n $wanted -t $QRAISE_TIME $QRAISE_CORES_FLAG $cores $QRAISE_EXTRA"
    # shellcheck disable=SC2086
    qraise -n "$wanted" -t "$QRAISE_TIME" $QRAISE_CORES_FLAG "$cores" $QRAISE_EXTRA || return 1
    trap drop_vqpus EXIT

    local waited=0
    while (( $(registered_vqpus) < wanted )); do
        if (( waited >= QRAISE_WAIT_S )); then
            echo "error: only $(registered_vqpus)/$wanted vQPUs registered after ${QRAISE_WAIT_S}s" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "$(registered_vqpus) vQPU(s) registered after ${waited}s"
    # SLURM job(s) behind these vQPUs (registry ids look like <jobid>_<pid>),
    # checked after the run for OOM kills -- see check_vqpu_jobs.
    RAISED_JOB_IDS=$(python3 -c "import json,sys; print(','.join(sorted({k.split('_')[0] for k in json.load(open(sys.argv[1]))})))" "$CUNQA_REGISTRY" 2>/dev/null || true)
    RAISED_COUNT="$wanted"
    RAISED_CORES="$cores"
    # Which core split each configuration used, for the analysis.
    [[ -f "$OUT_DIR/vqpu_cores.csv" ]] || echo "mode,vqpus,cores_per_vqpu,total_cores" > "$OUT_DIR/vqpu_cores.csv"
    echo "${CUNQA_PHASE:-unknown},$wanted,$cores,$(( wanted * cores ))" >> "$OUT_DIR/vqpu_cores.csv"
}

# Logs the SLURM state of the vQPU job(s) of the current configuration to
# <job>/vqpu_jobs.csv and warns about OOM kills. Rows marked "vqpu_lost" in
# the benchmark CSVs can be matched to their cause here.
check_vqpu_jobs() {
    [[ -n "${RAISED_JOB_IDS:-}" ]] || { echo "warning: unknown vQPU job id(s), cannot check for OOM kills" >&2; return 0; }
    local file="$OUT_DIR/vqpu_jobs.csv" lines
    [[ -f "$file" ]] || echo "mode,vqpus,cores_per_vqpu,job_step,state,exit_code,max_rss,req_mem" > "$file"
    lines=$(sacct -j "$RAISED_JOB_IDS" -n -P -o JobID,State,ExitCode,MaxRSS,ReqMem 2>/dev/null) || return 0
    while IFS='|' read -r step state code rss req; do
        [[ -n "$step" ]] && echo "${CUNQA_PHASE:-unknown},$RAISED_COUNT,$RAISED_CORES,$step,$state,$code,$rss,$req" >> "$file"
    done <<< "$lines"
    if grep -q "OUT_OF_ME" <<< "$lines"; then
        echo "WARNING: a vQPU of the $RAISED_COUNT x $RAISED_CORES-core configuration was OOM-killed" \
             "(see $file); rows after it are marked vqpu_lost or failed" >&2
    fi
}

# with_vqpus <count> <command...>: raise, run, drop. With CUNQA_RAISE=0 it just runs.
with_vqpus() {
    local k="$1"; shift
    raise_vqpus "$k" || { echo "error: could not raise $k vQPU(s), skipping this configuration" >&2; drop_vqpus; return 1; }
    "$@"
    local rc=$?
    check_vqpu_jobs
    drop_vqpus
    trap - EXIT
    return $rc
}

run_cunqa() {
    section "CUNQA emulator -- vQPU sweep: $CUNQA_VQPUS ($CUNQA_NODE_CORES cores in total each)"
    CUNQA_PHASE=vqe
    local vqe_args=("${COMMON[@]}" --devices cunqa --optimizers cpu
                    --min-qubits "$CUNQA_MIN_Q" --max-qubits "$CUNQA_MAX_Q" --step "$CUNQA_STEP_Q"
                    --csv "$OUT_DIR/cunqa.csv")
    : > "$OUT_DIR/cunqa.log"

    if [[ "$CUNQA_RAISE" != "1" ]]; then
        # Pre-raised vQPUs: core split isn't controlled here, sweep them in one go.
        echo "warning: CUNQA_RAISE=0, cores per vQPU are whatever the existing vQPUs were raised with" >&2
        "$BIN" "${vqe_args[@]}" --methods native,host --vqpus "$(echo "$CUNQA_VQPUS" | tr ' ' ',')" \
            2>&1 | tee -a "$OUT_DIR/cunqa.log"
        "$BIN" "${vqe_args[@]}" --methods sequential --vqpus 1 2>&1 | tee -a "$OUT_DIR/cunqa.log"
        return
    fi

    local k seq_done=0
    for k in $CUNQA_VQPUS; do
        # Sequential (one blocking circuit at a time) can't use more than one
        # vQPU, so it only runs in the 1-vQPU configuration.
        local methods=native,host
        if (( k == 1 )); then methods=native,host,sequential; seq_done=1; fi
        with_vqpus "$k" "$BIN" "${vqe_args[@]}" --methods "$methods" --vqpus "$k" \
            > >(tee -a "$OUT_DIR/cunqa.log") 2>&1
    done
    if (( seq_done == 0 )); then
        with_vqpus 1 "$BIN" "${vqe_args[@]}" --methods sequential --vqpus 1 \
            > >(tee -a "$OUT_DIR/cunqa.log") 2>&1
    fi
}

# ------------------------------------------------------------------ QMIO ----
run_qmio() {
    section "QMIO real QPU"
    mkdir -p "${STORE:-$HOME}/.qpu"
    echo "${ZMQ_SERVER:-}" > "${STORE:-$HOME}/.qpu/zmq_server.txt"

    # native and host only: for QMIO the batched path already submits one
    # circuit at a time, so "sequential" would just repeat "host". No warm-up:
    # every iteration is real QPU time.
    "$BIN" "${COMMON[@]}" --devices qmio --optimizers cpu \
        --methods native,host --iters "$QMIO_ITERS" --warmup 0 \
        --min-qubits "$QMIO_MIN_Q" --max-qubits "$QMIO_MAX_Q" --step "$QMIO_STEP_Q" \
        --csv "$OUT_DIR/qmio.csv" 2>&1 | tee "$OUT_DIR/qmio.log"

    echo "" > "${STORE:-$HOME}/.qpu/zmq_server.txt"
}

# -------------------------------------------------------- single circuit ----
run_circuit() {
    section "Circuits (no VQE) -- devices: $CIRCUIT_DEVICES, modes: $CIRCUIT_MODES"
    local circ_args=(--circuits "$CIRCUIT_KINDS"
                     --min-qubits "$CIRCUIT_MIN_Q" --max-qubits "$CIRCUIT_MAX_Q" --step "$CIRCUIT_STEP_Q"
                     --reps "$REPS" --shots "$SHOTS" --repeats "$CIRCUIT_REPEATS"
                     --time-limit "$CIRCUIT_TIME_LIMIT"
                     --batch-sizes "$CIRCUIT_BATCH_SIZES" --throughput-qubits "$CIRCUIT_TP_QUBITS"
                     --throughput-csv "$OUT_DIR/circuit_throughput.csv" --csv "$OUT_DIR/circuit.csv")
    : > "$OUT_DIR/circuit.log"

    # Every device except CUNQA: one run per device, each with its own memory guard
    # (the CSVs are appended to, so the rows end up in the same files).
    local dev
    for dev in $(echo "$CIRCUIT_DEVICES" | tr ',' ' '); do
        [[ "$dev" == "cunqa" ]] && continue
        if [[ "$dev" == "qmio" ]]; then
            mkdir -p "${STORE:-$HOME}/.qpu"
            echo "${ZMQ_SERVER:-}" > "${STORE:-$HOME}/.qpu/zmq_server.txt"
        fi
        "$CIRCUIT_BIN" "${circ_args[@]}" --max-memory-gb "$(mem_guard_gb "$dev")" \
            --devices "$dev" --modes "$CIRCUIT_MODES" 2>&1 | tee -a "$OUT_DIR/circuit.log"
    done

    # CUNQA: each vQPU count raised on its own, always CUNQA_NODE_CORES in total.
    if [[ ",$CIRCUIT_DEVICES," == *",cunqa,"* ]]; then
        CUNQA_PHASE=circuit
        local latency=0 throughput=0
        [[ ",$CIRCUIT_MODES," == *",latency,"* ]] && latency=1
        [[ ",$CIRCUIT_MODES," == *",throughput,"* ]] && throughput=1

        if [[ "$CUNQA_RAISE" != "1" ]]; then
            echo "warning: CUNQA_RAISE=0, cores per vQPU are whatever the existing vQPUs were raised with" >&2
            "$CIRCUIT_BIN" "${circ_args[@]}" --devices cunqa --modes "$CIRCUIT_MODES" \
                --vqpus "$(echo "$CIRCUIT_VQPUS" | tr ' ' ',')" 2>&1 | tee -a "$OUT_DIR/circuit.log"
            return
        fi

        local k lat_done=0
        if (( throughput )); then
            for k in $CIRCUIT_VQPUS; do
                # Latency needs a single vQPU (with all the cores): reuse the 1-vQPU raise.
                local modes=throughput
                if (( k == 1 && latency )); then modes=latency,throughput; lat_done=1; fi
                with_vqpus "$k" "$CIRCUIT_BIN" "${circ_args[@]}" --devices cunqa --modes "$modes" --vqpus "$k" \
                    > >(tee -a "$OUT_DIR/circuit.log") 2>&1
            done
        fi
        if (( latency && ! lat_done )); then
            with_vqpus 1 "$CIRCUIT_BIN" "${circ_args[@]}" --devices cunqa --modes latency \
                > >(tee -a "$OUT_DIR/circuit.log") 2>&1
        fi
    fi
}

# ------------------------------------------------------------------ main ----
case "$MODE" in
    cpu)     run_cpu ;;
    gpu)     run_gpu ;;
    cunqa)   run_cunqa ;;
    qmio)    run_qmio ;;
    circuit) run_circuit ;;
    all)     if [[ "$LOCAL" == "1" ]]; then run_cpu; run_gpu; run_circuit
             else run_cpu; run_cunqa; run_circuit; fi ;;
    *)       echo "usage: sbatch scripts/benchmarking.sh [cpu|gpu|cunqa|qmio|circuit|all]   (locally: scripts/benchmarking.sh ...)" >&2; exit 1 ;;
esac

section "Benchmarking done -- results in $OUT_DIR"
