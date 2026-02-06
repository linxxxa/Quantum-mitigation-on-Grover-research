import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# --- КОНСТАНТЫ ИССЛЕДОВАНИЯ ---
SCALE_1 = 1.0
SCALE_2 = 3.0
SHOTS = 10000
NUM_TRIALS = 5
TARGET_QUBITS = [0, 1, 2]

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
img_path = os.path.join(desktop, "brisbane_pure_science.png")

backend = FakeBrisbane()
noise_sim = AerSimulator.from_backend(backend)

def fold_manually(qc, scale=3):
    """
    Локальное свертывание гейтов (U -> U^scale). 
    Для гейтов H и CZ (где U^2 = I) это эквивалентно масштабированию шума.
    """
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in ['barrier', 'measure']:
            # Для scale=3 мы добавляем еще 2 выполнения гейта (всего 3)
            for _ in range(int(scale - 1)):
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded

def add_dd_manual(qc):
    """Стандартная защита X-X (Spin Echo) перед измерением"""
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        if inst.operation.name == 'measure':
            new_qc.barrier()
            for i in range(3): 
                new_qc.x(i); new_qc.x(i)
            new_qc.barrier()
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
    return new_qc

def richardson_extrapolation(p1, p2, s1, s2):
    """
    Формула линейной экстраполяции Ричардсона к нулевому шуму.
    P(0) = P(s1) + (P(s1) - P(s2)) / (s2 - s1) * s1
    """
    return p1 + (p1 - p2) / (s2 - s1) * s1

def run_experiment():
    # Базовая схема
    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.cz(0, 1) 
    qc.h(range(3))
    qc.measure_all()
    
    t_raw = transpile(qc, backend, initial_layout=TARGET_QUBITS, optimization_level=3)
    
    # 1. RAW (Lambda = 1)
    p_s1 = noise_sim.run(t_raw, shots=SHOTS).result().get_counts().get('111', 0) / SHOTS
    
    # 2. FOLDED (Lambda = 3)
    t_s3 = fold_manually(t_raw, scale=SCALE_2)
    p_s3 = noise_sim.run(t_s3, shots=SHOTS).result().get_counts().get('111', 0) / SHOTS
    
    p_zne = richardson_extrapolation(p_s1, p_s3, SCALE_1, SCALE_2)
    
    # 3. DD
    t_dd = add_dd_manual(t_raw)
    p_dd = noise_sim.run(t_dd, shots=SHOTS).result().get_counts().get('111', 0) / SHOTS
    
    # 4. HYBRID (ZNE поверх DD)
    t_dd_s3 = fold_manually(t_dd, scale=SCALE_2)
    p_dd_s3 = noise_sim.run(t_dd_s3, shots=SHOTS).result().get_counts().get('111', 0) / SHOTS
    p_hybrid = richardson_extrapolation(p_dd, p_dd_s3, SCALE_1, SCALE_2)
    
    return [p_s1, p_zne, p_dd, p_hybrid]

# --- ИСПОЛНЕНИЕ ---
print(f">>> Запуск {NUM_TRIALS} научных тестов для Brisbane...")
results = np.array([run_experiment() for _ in range(NUM_TRIALS)])

means = np.mean(results, axis=0)
errors = np.std(results, axis=0) 


labels = ['Raw', 'ZNE (L=3)', 'DD (X-X)', 'Hybrid']
plt.figure(figsize=(10, 6))
colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']

plt.bar(labels, means, yerr=errors, capsize=8, color=colors, edgecolor='black', zorder=3)
plt.axhline(y=0.125, color='red', linestyle='--', label='Random Guess (0.125)', zorder=4)

plt.title('Quantum Error Mitigation: Brisbane 127Q Analysis', fontsize=14)
plt.ylabel('P(111) Success Probability', fontsize=12)
plt.ylim(0, max(means + errors) * 1.3)
plt.grid(axis='y', alpha=0.3)
plt.legend()

plt.savefig(img_path, dpi=150)
print(f">>> Средние значения: {means}")
print(f">>> Ошибки: {errors}")