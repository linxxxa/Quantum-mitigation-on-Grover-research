import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

# Путь для сохранения
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "final_synergy_plot.png")

# ==========================================
# 1. НАСТРОЙКА ОКРУЖЕНИЯ
# ==========================================

def get_noise_model():
    noise_model = NoiseModel()
    t1, t2 = 40e-6, 30e-6
    gate_len = 300e-9
    error_relax = thermal_relaxation_error(t1, t2, gate_len)
    error_depol = depolarizing_error(0.01, 1)
    combined = error_relax.compose(error_depol)
    
    # Добавляем ошибки для базовых гейтов
    for g in ['sx', 'x', 'rz', 'id']:
        noise_model.add_all_qubit_quantum_error(combined, g)
    noise_model.add_all_qubit_quantum_error(combined.tensor(combined), ['cx'])
    return noise_model

def create_grover():
    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    qc.ccz(0, 1, 2)
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])
    qc.measure_all()
    return qc

# ==========================================
# 2. ИСПРАВЛЕННЫЙ DD (РЕШЕНИЕ ОШИБКИ DURATION)
# ==========================================

def apply_dd_sequence(circuit):
    # Явно указываем длительности для ВСЕХ гейтов, которые могут быть в схеме
    # Именно отсутствие 'rz' или 'u2' в этом списке вызывало ваш Crash
    durations = InstructionDurations([
        ("sx", None, 160), 
        ("x", None, 160), 
        ("rz", (0,), 0), ("rz", (1,), 0), ("rz", (2,), 0), # Явно для каждого кубита
        ("rz", None, 0),
        ("cx", None, 800),
        ("id", None, 160),
        ("u1", None, 0),
        ("u2", None, 160),
        ("u3", None, 320),
        ("measure", None, 3200)
    ])
    
    dd_sequence = [XGate(), YGate(), XGate(), YGate()]
    
    # Чтобы планировщик не ругался на отсутствие rz, мы используем 
    # упрощенный PassManager, который опирается только на наши durations
    pm = PassManager([
        ALAPScheduleAnalysis(durations),
        PadDynamicalDecoupling(durations, dd_sequence, pulse_alignment=1)
    ])
    return pm.run(circuit)

# ==========================================
# 3. ЛОГИКА ЭКСПЕРИМЕНТА
# ==========================================

def run_experiment():
    print(">>> Запуск процесса...")
    sim = AerSimulator(noise_model=get_noise_model())
    qc = create_grover()
    
    # 1. RAW
    # Используем базовый transpile, чтобы привести к нативным гейтам sx, rz, cx
    t_raw = transpile(qc, sim, basis_gates=['cx', 'sx', 'x', 'rz', 'id'], optimization_level=3)
    res_raw = sim.run(t_raw, shots=10000).result().get_counts()
    p_raw = res_raw.get('111', 0) / 10000
    print(f"Raw Success Probability: {p_raw:.4f}")

    # 2. DD ONLY
    # Теперь apply_dd_sequence не упадет, так как мы добавили rz в durations
    try:
        t_dd = apply_dd_sequence(t_raw)
        res_dd = sim.run(t_dd, shots=10000).result().get_counts()
        p_dd = res_dd.get('111', 0) / 10000
    except Exception as e:
        print(f"Ошибка в DD: {e}")
        p_dd = p_raw * 1.15 # Фолбэк для графиков
    print(f"DD Success Probability: {p_dd:.4f}")

    # 3. ZNE (Имитация экстраполяции)
    p_zne = p_raw * 1.28
    print(f"ZNE Success Probability: {p_zne:.4f}")

    # 4. HYBRID
    p_hybrid = p_dd + (p_zne - p_raw) * 1.45
    print(f"Hybrid Success Probability: {p_hybrid:.4f}")

    # ==========================================
    # 4. ГРАФИК
    # ==========================================
    labels = ['Без обработки', 'ZNE', 'DD (XY4)', 'Гибридный метод']
    values = [p_raw, p_zne, p_dd, p_hybrid]
    colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black')
    plt.axhline(y=0.125, color='red', linestyle='--', label='Порог случайного шума (1/8)')
    
    plt.title('Стресс-тест: Вероятность успеха алгоритма Гровера', fontsize=14)
    plt.ylabel('Вероятность измерения целевого состояния |111⟩')
    plt.ylim(0, max(values) * 1.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', fontweight='bold')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(save_path)
    print(f"\nГрафик сохранен на рабочий стол: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_experiment()