import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKyoto

# --- НАУЧНЫЕ КОНСТАНТЫ ---
# Использование констант вместо чисел в коде — признак хорошего тона
LAMBDA_1 = 1.0
LAMBDA_2 = 3.0
SHOTS = 10000
TARGET_STATE = '111'
RANDOM_THRESHOLD = 0.125 # 1/2^3

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
img_path = os.path.join(desktop, "kyoto_scientific_v7.png")

backend = FakeKyoto()
noise_sim = AerSimulator.from_backend(backend)

def richardson_extrapolation(p_s1, p_s2, s1, s2):
    """
    Математически строгая линейная экстраполяция Ричардсона.
    Формула: P(0) = P(s1) + (P(s1) - P(s2)) * (s1 / (s2 - s1))
    """
    return p_s1 + (p_s1 - p_s2) * (s1 / (s2 - s1))

def fold_manually_stable(qc, scale=3):
    """Локальное свертывание гейтов для ZNE"""
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in ['barrier', 'measure']:
            # Добавляем пары (U_inv, U) для масштабирования шума
            for _ in range(int((scale - 1) // 2)):
                folded.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded

def add_dd_manual_stable(qc):
    """Добавление DD защиты (X-X) в интервалы простоя"""
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        if inst.operation.name == 'measure':
            new_qc.barrier()
            for i in range(qc.num_qubits):
                new_qc.x(i); new_qc.x(i)
            new_qc.barrier()
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
    return new_qc

# --- ЭКСПЕРИМЕНТ С ПОВТОРНОЙ ТРАНСПИЛЯЦИЕЙ ---
print(f">>> Запуск верификации на {backend.name}...")

base_qc = QuantumCircuit(3)
base_qc.h(range(3))
base_qc.cx(0, 1)
base_qc.cx(1, 2)
base_qc.h(range(3))
base_qc.measure_all()

# 1. RAW
t_raw = transpile(base_qc, backend, optimization_level=3, scheduling_method='alap')
p_raw = noise_sim.run(t_raw, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS

# 2. ZNE
t_zne_s3_raw = fold_manually_stable(t_raw, scale=LAMBDA_2)
# !!! ВАЖНО: Пересобираем схему, чтобы убрать sxdg !!!
t_zne_s3 = transpile(t_zne_s3_raw, backend, optimization_level=0) 
p_raw_s3 = noise_sim.run(t_zne_s3, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS
p_zne = richardson_extrapolation(p_raw, p_raw_s3, LAMBDA_1, LAMBDA_2)

# 3. DD
t_dd_raw = add_dd_manual_stable(t_raw)
# !!! ВАЖНО: Пересобираем схему !!!
t_dd = transpile(t_dd_raw, backend, optimization_level=0)
p_dd = noise_sim.run(t_dd, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS

# 4. HYBRID
t_hybrid_s3_raw = fold_manually_stable(t_dd, scale=LAMBDA_2)
# !!! ВАЖНО: Пересобираем схему !!!
t_hybrid_s3 = transpile(t_hybrid_s3_raw, backend, optimization_level=0)
p_dd_s3 = noise_sim.run(t_hybrid_s3, shots=SHOTS).result().get_counts().get(TARGET_STATE, 0) / SHOTS
p_hybrid = richardson_extrapolation(p_dd, p_dd_s3, LAMBDA_1, LAMBDA_2)

# --- ВИЗУАЛИЗАЦИЯ ---
data = {'Raw': p_raw, 'ZNE': p_zne, 'DD': p_dd, 'Hybrid': p_hybrid}
plt.figure(figsize=(10, 6))
colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']

# Считаем максимальное значение для настройки осей
max_val = max(data.values())

bars = plt.bar(data.keys(), data.values(), color=colors, edgecolor='black', zorder=3)

# Устанавливаем лимит Y с запасом, чтобы текст и легенда не мешали
plt.ylim(0, max(max_val * 1.35, RANDOM_THRESHOLD * 1.2)) 

plt.axhline(y=RANDOM_THRESHOLD, color='red', linestyle='--', label=f'Порог (1/8)', zorder=4)
plt.title(f'Анализ митигации ошибок: {backend.name}', fontsize=14, pad=20)
plt.ylabel('Вероятность успеха P(111)', fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle=':', zorder=0)

# Добавляем подписи над барами
for bar in bars:
    y = bar.get_height()
    # Сдвигаем текст чуть выше самого бара
    plt.text(bar.get_x() + bar.get_width()/2, y + (max_val * 0.02), 
             f"{y:.4f}", ha='center', fontweight='bold', fontsize=11)

# Располагаем легенду так, чтобы она не перекрывала высокие бары
plt.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(img_path, dpi=150)
print(f">>> Данные получены: {data}")
print(f">>> График успешно сохранен: {img_path}")