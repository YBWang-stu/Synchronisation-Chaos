"""洛伦兹双向耦合同步：将前面九步代码整合到一个文件。

依赖：NumPy、SciPy 和 Matplotlib（当前科学计算环境已经安装）。
在笔记本单元格中运行：
    %run "C:/Users/wbr123/Documents/ChatGPT/chaos/lorenz_numerics.py"

也可以在科学计算环境的命令行中运行：python lorenz_numerics.py
所有输出保存在本文件旁的 output/lorenz_combined/ 中。
完整扫描需要等待几分钟，控制台会逐行打印进度。

学习顺序与原来的九个文件对应，使用单元格标记分段。
只想先看示例图时，把下面两个扫描开关改为关闭状态。
本文件自带全部方程和函数，不需要前面九个文件或预先计算的结果。
"""

# %% 公共设置：修改实验参数时从这里开始
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.integrate import solve_ivp

IN_JUPYTER = "ipykernel" in sys.modules
if not IN_JUPYTER:
    matplotlib.use("Agg")  # 命令行运行时只保存图片，不弹出窗口。
import matplotlib.pyplot as plt

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
INITIAL_STATE = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
DT = 0.01                    # 保存结果的时间间隔；求解器内部自适应步长。
RTOL = 1e-8                  # 求解器的相对误差容许量。
ATOL = 1e-10                 # 求解器的绝对误差容许量。
SYNC_TOL = 1e-6              # 同步判据，与求解器的容差含义不同。
PLOT_FLOOR = 1e-14           # 仅用于对数图显示，不修改实际误差。
RUN_COARSE_SWEEP = True
RUN_REFINED_SWEEP = True
SHOW_FIGURES = IN_JUPYTER    # 在笔记本中显示图片，在命令行中只保存图片。

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output" / "lorenz_combined"
BLUE, ORANGE, GREEN = "#245A81", "#D47222", "#32846B"


# %% 第 1 步：单个洛伦兹系统，输出的是变化率
def lorenz(t, state):
    x, y, z = state
    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    return [dx, dy, dz]


# %% 第 2 步：式 (9.33) 的双向耦合，三个对应分量都参与耦合
def coupled_lorenz(t, state, c):
    x1, y1, z1, x2, y2, z2 = state
    dx1, dy1, dz1 = lorenz(t, [x1, y1, z1])
    dx2, dy2, dz2 = lorenz(t, [x2, y2, z2])
    return [
        dx1 + c * (x2 - x1),
        dy1 + c * (y2 - y1),
        dz1 + c * (z2 - z1),
        dx2 + c * (x1 - x2),
        dy2 + c * (y1 - y2),
        dz2 + c * (z1 - z2),
    ]


# %% 第 3 步：统一的求解函数，后面的示例和扫描都调用它
def simulate(c, end_time=50.0):
    """每次从预设初始状态重新开始，返回时间数组和六行状态数组。"""
    times = np.linspace(0.0, end_time, int(round(end_time / DT)) + 1)
    solution = solve_ivp(
        coupled_lorenz, (0.0, end_time), INITIAL_STATE,
        args=(float(c),), t_eval=times, method="RK45", rtol=RTOL, atol=ATOL,
    )
    if not solution.success:
        raise RuntimeError(f"c={c}: {solution.message}")
    return solution.t, solution.y


def save_figure(fig, filename):
    """统一保存图片；在笔记本环境中同时显示。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=180)
    print("Saved:", path, flush=True)
    if SHOW_FIGURES:
        # 直接显示刚保存的图片，即使之前使用过非交互绘图后端也能显示。
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    plt.close(fig)


def save_json(filename, data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


def experiment_settings():
    """随结果保存实验设置，方便复现。"""
    return {
        "sigma": SIGMA, "rho": RHO, "beta": BETA,
        "initial_state": INITIAL_STATE, "sample_interval": DT,
        "method": "RK45", "rtol": RTOL, "atol": ATOL,
        "tolerance": SYNC_TOL,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    }


# %% 第 4 步：画两条时间序列
def draw_time_series(ax, t, state):
    ax.plot(t, state[0], color=BLUE, linewidth=1.3, label=r"$x_1(t)$")
    ax.plot(t, state[3], color=ORANGE, linewidth=1.0, linestyle="--", label=r"$x_2(t)$")
    ax.set(xlabel="Time t", ylabel="x coordinate", xlim=(t[0], t[-1]))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")


def plot_time_series(t, state, c):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    draw_time_series(ax, t, state)
    ax.set_title(f"Two-way coupled Lorenz systems: c = {c}")
    fig.tight_layout()
    save_figure(fig, "step4_time_series_c0p1.png")


# %% 第 5 步：同一时刻的 x1 与 x2 对应图
def draw_coordinate_pairs(ax, state):
    x1, x2 = state[0], state[3]
    ax.scatter(x1, x2, s=5, alpha=0.4, color=BLUE,
               edgecolors="none", label="Same-time pairs")
    low, high = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    ax.plot([low, high], [low, high], "--", color=ORANGE, label=r"$x_2=x_1$")
    padding = max(0.05 * (high - low), 0.1)
    ax.set(xlabel=r"$x_1(t)$", ylabel=r"$x_2(t)$",
           xlim=(low - padding, high + padding), ylim=(low - padding, high + padding))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", markerscale=2)


def plot_coordinate_pairs(state, c):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_coordinate_pairs(ax, state)
    ax.set_title(f"Simultaneous coordinates: c = {c}")
    fig.tight_layout()
    save_figure(fig, "step5_x1_vs_x2_c0p1.png")


# %% 第 6 步：较强耦合的两种图，复用上面的画图函数
def plot_strong_coupling(t, state, c):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.65, 1]})
    draw_time_series(axes[0], t, state)
    draw_coordinate_pairs(axes[1], state)
    axes[0].set_title("Time series")
    axes[1].set_title("Simultaneous coordinates")
    fig.suptitle(f"Two-way coupled Lorenz systems: c = {c}")
    fig.tight_layout()
    save_figure(fig, "step6_stronger_coupling_c1p0.png")


# %% 第 7 步：三维同步误差及有限时间判据
def synchronization_error(state):
    x1, y1, z1, x2, y2, z2 = state
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def window_max_error(t, error, start, end):
    selected = (t >= start) & (t <= end)
    if not np.any(selected):
        raise ValueError("The assessment window contains no sample times.")
    return float(np.max(error[selected]))


def add_log_plot_note(fig):
    fig.text(0.5, 0.015,
             "Values below 1e-14 are drawn at 1e-14 only for display; calculations use the original errors.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1))


def analyze_demo_error(t, state, c):
    error = synchronization_error(state)
    max_error = window_max_error(t, error, 25.0, 50.0)
    print(f"c={c}: max E on [25, 50] = {max_error:.6e}; passes = {max_error < SYNC_TOL}", flush=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(25, 50, color="#DCECE7", alpha=0.55, label="Assessment window: 25 to 50")
    ax.semilogy(t, np.maximum(error, PLOT_FLOOR), color=BLUE, label="E(t)")
    ax.axhline(SYNC_TOL, color=ORANGE, linestyle="--", label=f"Tolerance: {SYNC_TOL:g}")
    ax.set(xlabel="Time t", ylabel="Synchronization error", xlim=(0, 50),
           ylim=(1e-15, 1e2), title=f"Full-state synchronization error: c = {c}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    add_log_plot_note(fig)
    save_figure(fig, "step7_sync_error_c1p0.png")
    save_json("step7_error_summary.json", {
        "settings": experiment_settings(), "c": c, "window": [25, 50],
        "max_error": max_error, "passes_criterion": max_error < SYNC_TOL,
    })
    return error


# %% 第 8 步：粗略扫描，每个 c 都使用相同的一对初始状态
def coarse_sweep():
    c_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    rows = []
    data = {"complete": False, "settings": {
        **experiment_settings(), "end_time": 300.0, "transient": 200.0,
        "c_values": c_values,
    }, "results": rows}
    print("  c       max E [200,300]     passes", flush=True)
    for c in c_values:
        t, state = simulate(c, end_time=300.0)
        error = synchronization_error(state)
        maximum = window_max_error(t, error, 200.0, 300.0)
        rows.append({"c": c, "max_error": maximum, "passes_criterion": maximum < SYNC_TOL})
        save_json("step8_coarse_sweep.json", data)
        print(f"{c:4.1f}       {maximum:12.6e}     {maximum < SYNC_TOL}", flush=True)
    data["complete"] = True
    save_json("step8_coarse_sweep.json", data)

    fig, ax = plt.subplots(figsize=(8, 5))
    for passed, color, label in [(False, ORANGE, "Does not pass criterion"),
                                  (True, BLUE, "Passes criterion")]:
        subset = [row for row in rows if row["passes_criterion"] == passed]
        if subset:
            ax.scatter([row["c"] for row in subset],
                       [max(row["max_error"], PLOT_FLOOR) for row in subset],
                       color=color, s=55, label=label, zorder=3)
    ax.set_yscale("log")
    ax.axhline(SYNC_TOL, color=GREEN, linestyle="--", label=f"Tolerance: {SYNC_TOL:g}")
    ax.set(xlabel="Coupling strength c", ylabel="Maximum error on [200, 300]",
           xlim=(-0.04, 1.04), ylim=(1e-15, 1e3), title="Coarse sweep: same initial pair for every c")
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.grid(alpha=0.2)
    ax.legend(loc="center right", fontsize=9)
    add_log_plot_note(fig)
    save_figure(fig, "step8_coarse_sweep.png")
    return rows


# %% 第 9 步：加密扫描，同时比较短期和长期观察窗口
def refined_sweep():
    c_values = np.round(np.linspace(0.40, 0.50, 11), 2)
    rows, traces = [], []
    data = {"complete": False, "settings": {
        **experiment_settings(), "end_time": 1000.0, "c_values": c_values.tolist(),
        "short_window": [200.0, 300.0], "long_window": [900.0, 1000.0],
    }, "results": rows}
    print("   c      max E [200,300]    pass      max E [900,1000]   pass", flush=True)
    for c in c_values:
        t, state = simulate(c, end_time=1000.0)
        error = synchronization_error(state)
        max_short = window_max_error(t, error, 200.0, 300.0)
        max_long = window_max_error(t, error, 900.0, 1000.0)
        # 记录浮点数完全相等的情况，避免把数值上的零当成数学证明。
        nonzero = np.flatnonzero(error > 0)
        zero_tail_start = None
        if len(nonzero) and nonzero[-1] < len(t) - 1:
            zero_tail_start = float(t[nonzero[-1] + 1])
        rows.append({
            "c": float(c), "max_error_short": max_short, "max_error_long": max_long,
            "passes_short": max_short < SYNC_TOL, "passes_long": max_long < SYNC_TOL,
            "exact_zero_samples": int(np.count_nonzero(error == 0)),
            "exact_zero_tail_start": zero_tail_start,
        })
        traces.append(error)
        save_json("step9_refined_sweep.json", data)
        print(f"{c:5.2f}      {max_short:12.5e}  {str(max_short < SYNC_TOL):5}"
              f"      {max_long:12.5e}  {str(max_long < SYNC_TOL):5}", flush=True)
    data["complete"] = True
    save_json("step9_refined_sweep.json", data)
    np.savez_compressed(OUTPUT_DIR / "step9_error_traces.npz", t=t, c=c_values, error=np.asarray(traces))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].semilogy(c_values, np.maximum([row["max_error_short"] for row in rows], PLOT_FLOOR),
                     "o--", color=ORANGE, label="Window: 200 to 300")
    axes[0].semilogy(c_values, np.maximum([row["max_error_long"] for row in rows], PLOT_FLOOR),
                     "s-", color=BLUE, label="Window: 900 to 1000")
    axes[0].set(xlabel="Coupling strength c", ylabel="Maximum synchronization error",
                title="Refined sweep: two observation windows")
    axes[0].set_xticks(c_values[::2])
    for target, color in [(0.44, "#9A5B20"), (0.45, "#C55055"), (0.46, BLUE), (0.47, GREEN)]:
        idx = int(np.argmin(abs(c_values - target)))
        axes[1].semilogy(t, np.maximum(traces[idx], PLOT_FLOOR), color=color,
                         linewidth=0.8, label=f"c = {target:.2f}")
    axes[1].set(xlabel="Time t", ylabel="Synchronization error E(t)",
                xlim=(0, 1000), title="Error histories near the transition")
    for ax in axes:
        ax.axhline(SYNC_TOL, color="#555555", linestyle=":", label=f"Tolerance: {SYNC_TOL:g}")
        ax.set_ylim(1e-15, 1e3)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    add_log_plot_note(fig)
    save_figure(fig, "step9_refined_sweep.png")

    # 只报告相邻采样点之间的判据变化，不进行插值或宣称精确阈值。
    for left, right in zip(rows[:-1], rows[1:]):
        if left["passes_long"] != right["passes_long"]:
            print(f"Finite-time criterion changes between c={left['c']:.2f} and c={right['c']:.2f}.", flush=True)
    return rows


# %% 主程序：按九个原文件的顺序运行，重复使用已经计算好的示例轨迹
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1] Single Lorenz derivatives:", lorenz(0.0, [1.0, 1.0, 1.0]), flush=True)
    print("[2] Coupled derivatives:", coupled_lorenz(0.0, INITIAL_STATE, 0.1), flush=True)

    print("[3] Solving the c=0.1 example...", flush=True)
    t_weak, state_weak = simulate(0.1)
    print("Time shape:", t_weak.shape, "State shape:", state_weak.shape, flush=True)
    print("[4-5] Plotting the c=0.1 example...", flush=True)
    plot_time_series(t_weak, state_weak, 0.1)
    plot_coordinate_pairs(state_weak, 0.1)

    print("[6] Solving and plotting the c=1.0 example...", flush=True)
    t_strong, state_strong = simulate(1.0)
    plot_strong_coupling(t_strong, state_strong, 1.0)
    print("[7] Measuring full-state synchronization error...", flush=True)
    error_strong = analyze_demo_error(t_strong, state_strong, 1.0)

    # 保存示例数组，之后画图时也可以直接读取，不必重新求解。
    np.savez_compressed(OUTPUT_DIR / "demo_trajectories.npz",
                        t=t_weak, weak=state_weak, strong=state_strong, error_strong=error_strong)
    all_results = {"weak": (t_weak, state_weak), "strong": (t_strong, state_strong)}
    if RUN_COARSE_SWEEP:
        print("[8] Running the coarse sweep...", flush=True)
        all_results["coarse"] = coarse_sweep()
    if RUN_REFINED_SWEEP:
        print("[9] Running the refined sweep...", flush=True)
        all_results["refined"] = refined_sweep()
    print("Finished. Output directory:", OUTPUT_DIR, flush=True)
    return all_results


if __name__ == "__main__":
    results = main()
