import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKyoto

# --- КОНФИГУРАЦИЯ ---
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
img_path = os.path.join(desktop, "synergy_final_v6.png")

backend = FakeKyoto()
noise_sim = AerSimulator.from_backend(backend)

def fold_manually_stable(qc, scale=3):
    """Стабильный фолдинг для ZNE"""
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in ['barrier', 'measure']:
            for _ in range(scale - 1):
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded

def add_dd_manual_stable(qc):
    """Добавление легкой DD защиты (X-X) перед измерениями"""
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
print(">>> Запуск финальной симуляции (V6)...")

base_qc = QuantumCircuit(3)
base_qc.h(range(3))
base_qc.cx(0, 1)
base_qc.cx(1, 2)
base_qc.h(range(3))
base_qc.measure_all()

# Опорная транспиляция
t_raw = transpile(base_qc, noise_sim, optimization_level=1)

# 1. RAW
p_raw = noise_sim.run(t_raw, shots=10000).result().get_counts().get('111', 0) / 10000

# 2. ZNE (Правильный вызов fold_manually_stable)
t_zne_s3 = fold_manually_stable(t_raw, scale=3)
p_raw_s3 = noise_sim.run(t_zne_s3, shots=10000).result().get_counts().get('111', 0) / 10000
p_zne = p_raw + (p_raw - p_raw_s3) * 0.5

# 3. DD (Правильный вызов без scale)
t_dd = add_dd_manual_stable(t_raw)
p_dd = noise_sim.run(t_dd, shots=10000).result().get_counts().get('111', 0) / 10000

# 4. HYBRID
t_hybrid_s3 = fold_manually_stable(t_dd, scale=3)
p_dd_s3 = noise_sim.run(t_hybrid_s3, shots=10000).result().get_counts().get('111', 0) / 10000
p_hybrid = p_dd + (p_dd - p_dd_s3) * 0.5

# --- РЕЗУЛЬТАТЫ ---
data = {'Гровер': p_raw, 'ZNE': p_zne, 'DD': p_dd, 'Гибрид': p_hybrid}
print(f"Результаты: {data}")

# Отрисовка
plt.figure(figsize=(10, 6))
colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']
bars = plt.bar(data.keys(), data.values(), color=colors, edgecolor='black')
plt.axhline(y=0.125, color='red', linestyle='--', alpha=0.5, label='Порог (1/8)')
plt.title('Квантовая митигация ошибок: Kyoto', fontsize=14)
plt.ylabel('Вероятность успеха P(111)')

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.005, f"{y:.4f}", ha='center', fontweight='bold')

plt.ylim(0, max(data.values()) * 1.3)
plt.legend()
plt.savefig(img_path)
print(">>> Успех! Файл сохранен.")