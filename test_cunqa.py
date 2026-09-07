import os, sys, json

# Adding path to access CUNQA module
os.getenv("HOME")

# Gettting the raised QPUs
from cunqa.qpu import get_QPUs
from cunqa.qjob import QJob

_original_submit = QJob.submit

def _submit_with_debug(self, param_values=None):
    print("=== OUTGOING QUANTUM_TASK ===")
    print(json.dumps(self._quantum_task, default=str, indent=2))
    print("==============================")
    return _original_submit(self, param_values)

QJob.submit = _submit_with_debug

qpus  = get_QPUs(co_located=True)

# Creating a circuit to run in our QPUs
from cunqa.circuit import CunqaCircuit

qc = CunqaCircuit(num_qubits = 2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()

# Submitting the same circuit to all vQPUs
from cunqa.qpu import run

qcs = [qc] * 4
qjobs = run(qcs , qpus, shots = 1000)

print(qjobs)

# Gathering results
from cunqa.qjob import gather

results = gather(qjobs)

# Getting and printing the counts
counts_list = [result.counts for result in results]

for counts in counts_list:
    print(f"Counts: {counts}" ) # Format: {'00':546, '11':454}