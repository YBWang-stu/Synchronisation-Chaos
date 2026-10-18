
from pathlib import Path
import os
import sys

import matplotlib
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, MaxNLocator, FuncFormatter
import numpy as np

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
t_end = 60.0
h = 0.01
message_start = 40.0
X1_0 = [1.0, 1.0, 1.0]
X2_0 = [2.0, 3.0, 4.0]
OUTPUT_DIR = Path(__file__).resolve().parent
CHECK_HALF_STEP = True
SHOW_PLOTS = plt.get_backend().lower() != "agg"

def configure_times_new_roman():
    # Prefer local fonts, then standard AutoDL/Linux and Windows locations.
    font_dirs = [
        OUTPUT_DIR / "fonts",
        Path("/usr/share/fonts/truetype/times-new-roman"),
        Path.home() / ".local/share/fonts/times-new-roman",
        Path("C:/Windows/Fonts"),
    ]
    for folder in font_dirs:
        for name in ("times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf"):
            path = folder / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman:italic",
        "mathtext.fallback": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "savefig.facecolor": "white",
    })
    if "mathtext.bfit" in plt.rcParams:
        plt.rcParams["mathtext.bfit"] = "Times New Roman:italic:bold"


def lorenz(X):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def message(time, A, f):
    if time < message_start:
        return 0.0
    return A * np.sin(2.0 * np.pi * f * (time - message_start))

def one_way_lorenz(time, U, A, f):
    X1 = U[:3]
    x1 = X1[0]
    x2, y2, z2 = U[3:]

    # The receiver only receives s, not the clean x1 or the message separately.
    s = x1 + message(time, A, f)

    # Extend the original x2_damping receiver using the transmitted signal s.
    dx2 = sigma * (y2 - x2)
    dy2 = s * (rho - z2) - y2
    dz2 = s * y2 - beta * z2
    dX2 = np.array([dx2, dy2, dz2])
    return np.concatenate([lorenz(X1), dX2])

def rk4_step(time, U, step, A, f):
    # Evaluate the time-dependent message at all four RK4 stage times.
    k1 = one_way_lorenz(time, U, A, f)
    k2 = one_way_lorenz(time + step / 2, U + step / 2 * k1, A, f)
    k3 = one_way_lorenz(time + step / 2, U + step / 2 * k2, A, f)
    k4 = one_way_lorenz(time + step, U + step * k3, A, f)
    return U + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(A, f, step=h):
    n_steps = int(round(t_end / step))
    t = np.arange(n_steps + 1) * step
    U = np.array(X1_0 + X2_0, dtype=float)
    history = np.empty((n_steps + 1, 6))
    history[0] = U
    for n in range(n_steps):
        U = rk4_step(t[n], U, step, A, f)
        history[n + 1] = U
    return t, history


def main():
    configure_times_new_roman()
    plt.rcParams.update({
        "font.size":16,"axes.labelsize":17,"axes.titlesize":18,
        "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,
    })

    # Verify synchronisation before adding a message.
    t0,U0=simulate(0.0,5.0)
    error=np.linalg.norm(U0[:,:3]-U0[:,3:],axis=1)
    maximum=np.max(error[t0>=40.0])
    print(f"No-message maximum full-state error: {maximum:.6e}")
    print(f"No-message synchronisation: {maximum<1e-5}\n")

    cases=[('a',0.1,5.0),('b',2.0,5.0),('c',0.1,0.1)]
    results=[]

    fig,axes=plt.subplots(3,3,figsize=(12.8,7.7),layout='constrained',sharex='col')
    fig.get_layout_engine().set(w_pad=.045,h_pad=.04,wspace=.045,hspace=.025)
    fmt=FuncFormatter(lambda x,pos: f'{x:g}')

    for col,(tag,A,f) in enumerate(cases):
        times,U=simulate(A,f)
        original=np.array([message(time,A,f) for time in times])
        transmitted=U[:,0]+original
        recovered=transmitted-U[:,3]
        window=times >= (58.0 if f>=1 else 50.0)
        tw=times[window]
        a0,a1,a2=axes[:,col]

        a0.plot(tw,original[window],color='black',lw=1.3)
        a1.plot(tw,transmitted[window],color='tab:blue',lw=1.3)
        l0,=a2.plot(tw,original[window],color='black',lw=1.4,label='Original')
        l1,=a2.plot(tw,recovered[window],color='tab:orange',ls='--',lw=1.4,label='Recovered')

        a0.set_title(rf'({tag}) $A={A:g},\ f={f:g}$',pad=7)
        a0.set_ylim(-1.22*A,1.22*A)
        a0.set_yticks([-A,0,A])
        a1.set_ylim(-23,23)
        a1.set_yticks([-20,0,20])
        maximum=max(A,np.max(np.abs(recovered[window])))
        a2.set_ylim(-1.15*maximum,1.15*maximum)
        if col<2:a2.set_yticks([-A,0,A])
        else:a2.yaxis.set_major_locator(MaxNLocator(nbins=3,symmetric=True))
        ticks=[58,58.5,59,59.5,60] if f>=1 else [50,52.5,55,57.5,60]
        a2.xaxis.set_major_locator(FixedLocator(ticks))
        for axis in (a0,a1,a2):
            axis.set_xlim(tw[0],tw[-1])
            axis.grid(alpha=.2,lw=.65)
            axis.tick_params(direction='out',length=3,pad=3)
            axis.xaxis.set_major_formatter(fmt)
            axis.yaxis.set_major_formatter(fmt)

        late=times>=50
        rel=np.sqrt(np.mean((recovered[late]-original[late])**2))/np.sqrt(np.mean(original[late]**2))
        corr=np.corrcoef(original[late],recovered[late])[0,1]
        rmse=np.sqrt(np.mean((recovered[late]-original[late])**2))
        fine_relative_error=np.nan
        if CHECK_HALF_STEP:
            tf,Uf=simulate(A,f,h/2)
            mf=np.array([message(time,A,f) for time in tf])
            recovered_fine=Uf[:,0]+mf-Uf[:,3]
            fine_window=tf>=50.0
            fine_rmse=np.sqrt(np.mean((recovered_fine[fine_window]-mf[fine_window])**2))
            fine_relative_error=fine_rmse/np.sqrt(np.mean(mf[fine_window]**2))
        results.append([A,f,rmse,rel,corr,fine_relative_error])
        print(f'({tag}) A={A:g}, f={f:g}, relative RMS={rel:.4%}, correlation={corr:.6f}')
        if CHECK_HALF_STEP:
            print(f'    Half-step relative RMS={fine_relative_error:.4%}')

    axes[0,0].set_ylabel('Original message\n'+r'$m(t)$')
    axes[1,0].set_ylabel('Transmitted signal\n'+r'$s(t)$')
    axes[2,0].set_ylabel('Message recovery\n'+r'$m(t),\ \hat m(t)$')
    fig.supxlabel(r'Time $t$',fontsize=17)
    fig.legend([l0,l1],['Original','Recovered'],loc='outside lower right',ncol=2,
               frameon=False,handlelength=1.7,columnspacing=1.0)


    path=OUTPUT_DIR/"figure3.pdf"
    fig.savefig(path,format="pdf",dpi=800,facecolor="white")
    np.savetxt(
        OUTPUT_DIR/"recovery_results.csv",results,delimiter=",",
        header="amplitude,frequency,rmse,rmse_over_message_rms,correlation,half_step_rmse_over_message_rms",
        comments="",
    )
    print(f"Saved: {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
