import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# --- КОНСТАНТЫ ИССЛЕДОВАНИЯ ---
LAMBDA_1 = 1.0
LAMBDA_2 = 3.0
SHOTS = 10000
TARGET_STATE = '111'

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
img_path = os.path.join(desktop, "sherbrooke_scientific_pure.png")

backend = FakeSherbrooke()
noise_sim = AerSimulator.from_backend(backend)

def richardson_extrapolation(p_s1, p_s2, s1, s2):
    """Строгая формула экстраполяции Ричардсона"""
    return p_s1 + (p_s1 - p_s2) * (s1 / (s2 - s1))

def fold_manually_scientific(qc, scale=3):
    """Научно-корректный фолдинг: замена U на U(U_inv U)^n"""
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in ['barrier', 'measure']:
            # Для scale=3 добавляем одну пару (U_dagger, U)
            n_pairs = int((scale - 1) // 2)
            for _ in range(n_pairs):
                folded.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded

def add_dd_scientific(qc):
    """Внедрение DD (X-X) перед измерением"""
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        if inst.operation.name == 'measure':
            new_qc.barrier()
            for i in range(qc.num_qubits):
                new_qc.x(i); new_qc.x(i)
            new_qc.barrier()
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
    return new_qc

# --- ЭКСПЕРИМЕНТ ---
print(f">>> Запуск исследования на {backend.name} (127Q)...")

qc = QuantumCircuit(3)
qc.h(range(3))
qc.cz(0, 1)
qc.cz(1, 2)
qc.h(range(3))
qc.measure_all()

# Базовая транспиляция (ALAP для DD)
t_raw = transpile(qc, backend, initial_layout=[0, 1, 2], 
                  optimization_level=3, scheduling_method='alap')

# 1. RAW
p_raw = noise_sim.run(t_raw, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS

# 2. ZNE
t_zne_raw = fold_manually_scientific(t_raw, scale=LAMBDA_2)
# Пересобираем схему (L=0), чтобы разложить инвертированные гейты (напр. SXdg) в базис
t_zne = transpile(t_zne_raw, backend, optimization_level=0)
p_s3 = noise_sim.run(t_zne, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS
p_zne = richardson_extrapolation(p_raw, p_s3, LAMBDA_1, LAMBDA_2)

# 3. DD
t_dd_raw = add_dd_scientific(t_raw)
t_dd = transpile(t_dd_raw, backend, optimization_level=0)
p_dd = noise_sim.run(t_dd, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS

# 4. HYBRID
t_hyb_raw = fold_manually_scientific(t_dd, scale=LAMBDA_2)
t_hybrid = transpile(t_hyb_raw, backend, optimization_level=0)
p_hyb_s3 = noise_sim.run(t_hybrid, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS
p_hybrid = richardson_extrapolation(p_dd, p_hyb_s3, LAMBDA_1, LAMBDA_2)

# --- ГРАФИК ---
data = {'Raw': p_raw, 'ZNE': p_zne, 'DD': p_dd, 'Hybrid': p_hybrid}
print(f"Результаты: {data}")

plt.figure(figsize=(10, 6))
colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']
bars = plt.bar(data.keys(), data.values(), color=colors, edgecolor='black', zorder=3)

plt.axhline(y=0.125, color='red', linestyle='--', label='Random Level (0.125)')
plt.title(f'Error Mitigation Case Study: {backend.name}', fontsize=14)
plt.ylabel('P(111) Success Probability')
plt.grid(axis='y', alpha=0.3, zorder=0)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.002, f"{y:.4f}", ha='center', fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig(img_path, dpi=150)
print(f">>> График сохранен: {img_path}")