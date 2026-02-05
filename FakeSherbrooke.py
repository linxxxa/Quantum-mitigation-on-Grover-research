import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke # Меняем на Sherbrooke

# --- НАСТРОЙКИ ---
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
img_path = os.path.join(desktop, "sherbrooke_synergy_final.png")

backend = FakeSherbrooke()
noise_sim = AerSimulator.from_backend(backend)

def fold_manually(qc, scale=3):
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in ['barrier', 'measure']:
            for _ in range(scale - 1):
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded

def add_dd_manual(qc):
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        if inst.operation.name == 'measure':
            new_qc.barrier()
            for i in range(3):
                new_qc.x(i)
                new_qc.x(i)
            new_qc.barrier()
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
    return new_qc

# --- ЭКСПЕРИМЕНТ ---
print(">>> Запуск симуляции на FakeSherbrooke (Оптимальный режим)...")

qc = QuantumCircuit(3)
qc.h(range(3))
qc.cz(0, 1)
qc.cz(1, 2)
qc.h(range(3))
qc.measure_all()

# Используем те же кубиты для чистоты эксперимента
t_raw = transpile(qc, backend, initial_layout=[0, 1, 2], optimization_level=3)

# 1. RAW
p_raw = noise_sim.run(t_raw, shots=10000).result().get_counts().get('111', 0) / 10000

# 2. ZNE
t_zne_s3 = fold_manually(t_raw, scale=3)
p_raw_s3 = noise_sim.run(t_zne_s3, shots=10000).result().get_counts().get('111', 0) / 10000
p_zne = p_raw + (p_raw - p_raw_s3) * 0.5

# 3. DD
t_dd = add_dd_manual(t_raw)
p_dd = noise_sim.run(t_dd, shots=10000).result().get_counts().get('111', 0) / 10000

# 4. HYBRID
t_hybrid_s3 = fold_manually(t_dd, scale=3)
p_dd_s3 = noise_sim.run(t_hybrid_s3, shots=10000).result().get_counts().get('111', 0) / 10000
p_hybrid = p_dd + (p_dd - p_dd_s3) * 0.5

# --- ГРАФИК ---
data = {'Гровер': p_raw, 'ZNE': p_zne, 'DD': p_dd, 'Hybrid': p_hybrid}
print(f"Результаты Sherbrooke: {data}")

plt.figure(figsize=(10, 6))
colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']
bars = plt.bar(data.keys(), data.values(), color=colors, edgecolor='black', width=0.6)

plt.axhline(y=0.125, color='black', linestyle=':', alpha=0.5, label='Порог (1/8)')
plt.title('Митигация ошибок на FakeSherbrooke (127Q)', fontsize=14)
plt.ylabel('Вероятность успеха P(111)')

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.005, f"{y:.4f}", ha='center', fontweight='bold')

plt.ylim(0, max(data.values()) * 1.3)
plt.legend()
plt.savefig(img_path, dpi=150)
print(f">>> График сохранен: {img_path}")