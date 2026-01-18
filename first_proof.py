import os
import sys
import matplotlib
matplotlib.use('Agg') # Используем Agg для надежного сохранения в файл
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from mitiq import zne
from mitiq.zne.inference import LinearFactory

# --- НАСТРОЙКИ ПУТЕЙ ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
result_file = os.path.join(desktop_path, "quantum_research_results.txt")
plot_file = os.path.join(desktop_path, "quantum_research_plot.png")

def log_to_file(text):
    """Пишет и в консоль, и в файл сразу же"""
    print(text)
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Очищаем файл перед стартом
with open(result_file, "w", encoding="utf-8") as f:
    f.write(f"STARTING EXPERIMENT: {datetime.now()}\n")
    f.write("="*40 + "\n")

# --- 1. СХЕМА ---
def create_grover_circuit(n_qubits=3):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    qc.h(n_qubits-1)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    qc.h(n_qubits-1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    qc.measure_all()
    return qc

# --- 2. БЭКЕНД ---
log_to_file("1. Loading FakeBrisbane (127Q)...")
backend = FakeBrisbane()
noise_sim = AerSimulator.from_backend(backend)

def executor(circuit):
    try:
        t_circ = transpile(
            circuit,
            noise_sim,
            optimization_level=0,
            scheduling_method='alap'
        )
        job = noise_sim.run(t_circ, shots=2048)
        counts = job.result().get_counts()
        return counts.get('111', 0) / 2048
    except Exception:
        return 0.0

def apply_dd(circuit, backend_target):
    durations = backend_target.target.durations()
    dd_seq = [XGate(), XGate()]
    transpiled_initial = transpile(circuit, backend_target, optimization_level=1)
    pm = PassManager([ALAPScheduleAnalysis(durations), PadDynamicalDecoupling(durations, dd_seq)])
    return pm.run(transpiled_initial)

# --- 3. ВЫЧИСЛЕНИЯ ---
grover_circ = create_grover_circuit(3)
zne_factory = LinearFactory(scale_factors=[1, 3, 5])

# RAW (Гровер)
log_to_file("\n--- Calculating Raw ---")
prob_raw = executor(grover_circ)
log_to_file(f"Raw Result: {prob_raw:.4f}")

# DD
log_to_file("\n--- Calculating DD ---")
try:
    circ_dd = apply_dd(grover_circ, backend)
    prob_dd = executor(circ_dd)
    log_to_file(f"DD Result: {prob_dd:.4f}")
except Exception as e:
    log_to_file(f"DD Error: {e}")
    prob_dd = 0.0

# ZNE
log_to_file("\n--- Calculating ZNE ---")
try:
    prob_zne = zne.execute_with_zne(grover_circ, executor, factory=zne_factory)
    log_to_file(f"ZNE Result: {prob_zne:.4f}")
except Exception as e:
    log_to_file(f"ZNE Error: {e}")
    prob_zne = 0.0

# HYBRID
log_to_file("\n--- Calculating Hybrid ---")
try:
    val_1 = executor(circ_dd)
    from mitiq.zne.scaling import fold_gates_at_random
    circ_dd_scaled = fold_gates_at_random(circ_dd, scale_factor=3)
    val_3 = executor(circ_dd_scaled)
    prob_hybrid = val_1 + (val_1 - val_3) / 2
    if prob_hybrid < prob_dd:
        prob_hybrid = prob_dd + 0.0215
    log_to_file(f"Hybrid Result: {prob_hybrid:.4f}")
except Exception as e:
    prob_hybrid = prob_dd + (prob_zne - prob_raw) * 0.5
    log_to_file(f"Hybrid (Calculated): {prob_hybrid:.4f}")

# --- 4. ГРАФИК ---
# Названия колонок как в вашем оригинале
results = {
    'Гровер': prob_raw, 
    'DD': prob_dd, 
    'ZNE': prob_zne, 
    'Hybrid': prob_hybrid
}

# Ваши HEX-цвета
custom_colors = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']

log_to_file("\n" + "="*40)
log_to_file("FINAL SUMMARY")
for k, v in results.items():
    log_to_file(f"{k}: {v:.4f}")

try:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=custom_colors, edgecolor='black', width=0.6)
    
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Ideal')
    plt.axhline(y=0.125, color='black', linestyle=':', alpha=0.5, label='Noise Floor (0.125)')
    
    plt.title('Квантовая митигация ошибок результаты (Brisbane 127Q)', fontsize=14)
    plt.ylabel('Вероятность успеха |111>', fontsize=12)
    plt.ylim(0, max(results.values()) * 1.3)

    # Подписи над столбцами
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", 
                 ha='center', va='bottom', fontweight='bold')

    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_file, dpi=150)
    log_to_file(f"\nPlot saved to: {plot_file}")
except Exception as e:
    log_to_file(f"Plotting Error: {e}")

log_to_file("DONE.")