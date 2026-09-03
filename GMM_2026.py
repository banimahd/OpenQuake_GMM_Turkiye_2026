"""
GMM TURKIYE 2026 - GUI APPLICATION
===================================
Ground Motion Model for Turkiye (2026) with PINN Implementation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.io as sio
import os
import warnings

warnings.filterwarnings('ignore')


class GMMPredictor:
    """GMM Turkiye 2026 Predictor using extracted weights"""

    def __init__(self, weights_file='weights_GMM_Turkiye_2026.mat'):
        """Initialize the predictor by loading weights from MATLAB file."""
        print("=" * 60)
        print("GMM TURKIYE 2026 PREDICTOR")
        print("=" * 60)
        print(f"\nLoading weights from: {weights_file}")

        self.data = sio.loadmat(weights_file)
        self.weights = self.data["weights_python"][0]

        if "Param_Max_save" in self.data:
            self.param_max = self.data["Param_Max_save"].flatten()
            self.param_min = self.data["Param_Min_save"].flatten()
        elif "Param_Max" in self.data:
            self.param_max = self.data["Param_Max"].flatten()
            self.param_min = self.data["Param_Min"].flatten()
        else:
            print("Param_Max/Param_Min not found. Using default values.")
            self.param_max = np.array([100, 3, 8, 200, 1500])
            self.param_min = np.array([0, 1, 4, 0, 100])

        print(f"Loaded {len(self.weights)} networks")
        print(f"Input parameters: {len(self.param_min)}")

        net0 = self.weights[0]
        w1 = net0["fc1_Weights"][0, 0]  # type: ignore
        w2 = net0["fc5_Weights"][0, 0]  # type: ignore

        print(f"Network architecture:")
        print(f"  Input:  {w1.shape[1]} parameters")
        print(f"  Hidden: {w1.shape[0]} neurons")
        print(f"  Output: {w2.shape[0]} parameters")
        print("=" * 60)

    @staticmethod
    def mapminmax_apply(x, xmin, xmax, ymin=-1, ymax=1):
        """Apply mapminmax preprocessing (scales to [-1, 1])"""
        range_x = xmax - xmin
        range_x[range_x == 0] = 1e-10
        if np.isscalar(range_x) or range_x.size == 1:
            range_x = max(range_x, 1e-10)
        else:
            range_x[range_x == 0] = 1e-10
        return (ymax - ymin) * (x - xmin) / range_x + ymin

    @staticmethod
    def mapminmax_reverse(y, xmin, xmax, ymin=-1, ymax=1):
        """Reverse mapminmax preprocessing"""
        range_x = xmax - xmin
        range_x[range_x == 0] = 1e-10
        if np.isscalar(range_x) or range_x.size == 1:
            range_x = max(range_x, 1e-10)
        else:
            range_x[range_x == 0] = 1e-10
        return (y - ymin) * range_x / (ymax - ymin) + xmin

    def normalize_input(self, input_raw):
        """Normalize input using GMM formula"""
        range_x = self.param_max - self.param_min
        range_x[range_x == 0] = 1e-10
        return 0.60 * (input_raw - self.param_min.reshape(-1, 1)) / range_x.reshape(-1, 1) + 0.20

    def predict(self, fd, fm, mw, rjb, vs30):
        """Predict ground motion parameters."""
        input_raw = np.array([fd, fm, mw, rjb, vs30], dtype=np.float64).reshape(-1, 1)
        input_norm = self.normalize_input(input_raw)

        predictions = []

        for net in self.weights:
            try:
                w1 = net["fc1_Weights"][0, 0]  # type: ignore
                b1 = net["fc1_Bias"][0, 0]      # type: ignore
                w2 = net["fc5_Weights"][0, 0]  # type: ignore
                b2 = net["fc5_Bias"][0, 0]      # type: ignore

                w1 = np.array(w1, dtype=np.float64)
                b1 = np.array(b1, dtype=np.float64).reshape(-1, 1)
                w2 = np.array(w2, dtype=np.float64)
                b2 = np.array(b2, dtype=np.float64).reshape(-1, 1)

                xmin = np.array(net["input_xmin"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)  # type: ignore
                xmax = np.array(net["input_xmax"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)  # type: ignore
                ymin = np.array(net["input_ymin"][0, 0], dtype=np.float64).flatten()                # type: ignore
                ymax = np.array(net["input_ymax"][0, 0], dtype=np.float64).flatten()                # type: ignore

                xmin_out = np.array(net["output_xmin"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)  # type: ignore
                xmax_out = np.array(net["output_xmax"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)  # type: ignore
                ymin_out = np.array(net["output_ymin"][0, 0], dtype=np.float64).flatten()                # type: ignore
                ymax_out = np.array(net["output_ymax"][0, 0], dtype=np.float64).flatten()                # type: ignore

                input_proc = self.mapminmax_apply(input_norm, xmin, xmax, ymin, ymax)
                hidden = np.tanh(w1 @ input_proc + b1)
                output_proc = w2 @ hidden + b2
                output = self.mapminmax_reverse(output_proc, xmin_out, xmax_out, ymin_out, ymax_out)

                predictions.append(output.flatten())

            except Exception:
                continue

        if not predictions:
            return self._get_default_outputs()

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)

        results = {
            'PGA': round(np.exp(mean_pred[0]) * 986, 2),
            'PGV': round(np.exp(mean_pred[1]), 2),
            'Ia': round(np.exp(mean_pred[2]), 3),
            'D_5_75': round(np.exp(mean_pred[3]), 2),
            'D_5_95': round(np.exp(mean_pred[4]), 2),
            'T_m': round(np.exp(mean_pred[5]), 2),
            'CAV': round(np.exp(mean_pred[6]), 2),
            'PSa_003sec': round(np.exp(mean_pred[7]) * 986, 2),
            'PSa_005sec': round(np.exp(mean_pred[8]) * 986, 2),
            'PSa_0075sec': round(np.exp(mean_pred[9]) * 986, 2),
            'PSa_01sec': round(np.exp(mean_pred[10]) * 986, 2),
            'PSa_015sec': round(np.exp(mean_pred[11]) * 986, 2),
            'PSa_02sec': round(np.exp(mean_pred[12]) * 986, 2),
            'PSa_025sec': round(np.exp(mean_pred[13]) * 986, 2),
            'PSa_03sec': round(np.exp(mean_pred[14]) * 986, 2),
            'PSa_04sec': round(np.exp(mean_pred[15]) * 986, 2),
            'PSa_05sec': round(np.exp(mean_pred[16]) * 986, 2),
            'PSa_075sec': round(np.exp(mean_pred[17]) * 986, 2),
            'PSa_10sec': round(np.exp(mean_pred[18]) * 986, 2),
            'PSa_15sec': round(np.exp(mean_pred[19]) * 986, 2),
            'PSa_20sec': round(np.exp(mean_pred[20]) * 986, 2),
            'PSa_25sec': round(np.exp(mean_pred[21]) * 986, 2),
            'PSa_30sec': round(np.exp(mean_pred[22]) * 986, 2),
            'PSa_35sec': round(np.exp(mean_pred[23]) * 986, 2),
            'PSa_40sec': round(np.exp(mean_pred[24]) * 986, 2),
        }

        return results

    @staticmethod
    def _get_default_outputs():
        """Return default outputs when weights are not available."""
        return {
            'PGA': 0.0, 'PGV': 0.0, 'Ia': 0.0,
            'D_5_75': 0.0, 'D_5_95': 0.0, 'T_m': 0.0, 'CAV': 0.0,
            'PSa_003sec': 0.0, 'PSa_005sec': 0.0, 'PSa_0075sec': 0.0,
            'PSa_01sec': 0.0, 'PSa_015sec': 0.0, 'PSa_02sec': 0.0,
            'PSa_025sec': 0.0, 'PSa_03sec': 0.0, 'PSa_04sec': 0.0,
            'PSa_05sec': 0.0, 'PSa_075sec': 0.0, 'PSa_10sec': 0.0,
            'PSa_15sec': 0.0, 'PSa_20sec': 0.0, 'PSa_25sec': 0.0,
            'PSa_30sec': 0.0, 'PSa_35sec': 0.0, 'PSa_40sec': 0.0,
        }


class GMMApp:
    """Main GUI Application for GMM Turkiye 2026"""

    def __init__(self, root):
        self.root = root
        self.root.title("GMM Turkiye 2026 - Physics-informed Ground Motion Model")
        self.root.geometry("1150x950")
        self.root.minsize(1050, 880)
        self.root.configure(bg='#f0f2f5')

        self.font_family = 'Times New Roman'

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11
        plt.rcParams['mathtext.fontset'] = 'stix'

        try:
            self.predictor = GMMPredictor('weights_GMM_Turkiye_2026.mat')
        except Exception:
            print("Error loading predictor")
            self.predictor = None

        self.mw = tk.DoubleVar(value=7.0)
        self.rjb = tk.DoubleVar(value=10.0)
        self.vs30 = tk.DoubleVar(value=360.0)
        self.fd = tk.DoubleVar(value=3.0)
        self.fm_str = tk.StringVar(value='Reverse')

        self.pga = tk.StringVar(value="0")
        self.pgv = tk.StringVar(value="0")
        self.ia = tk.StringVar(value="0")
        self.d575 = tk.StringVar(value="0")
        self.d595 = tk.StringVar(value="0")
        self.tm = tk.StringVar(value="0")
        self.cav = tk.StringVar(value="0")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()

        self.create_widgets()
        self.plot_initial()

    def create_widgets(self):
        """Create all GUI widgets."""
        main = tk.Frame(self.root, bg='#f0f2f5')
        main.pack(fill='both', expand=True, padx=15, pady=10)

        top_frame = tk.Frame(main, bg='#f0f2f5')
        top_frame.pack(fill='x', pady=(0, 10))

        in_frame = tk.LabelFrame(top_frame, text=" INPUTS ",
                                 font=(self.font_family, 12, 'bold'),
                                 bg='#f0f2f5', fg='#2c3e50', padx=10, pady=8)
        in_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        input_grid = tk.Frame(in_frame, bg='#f0f2f5')
        input_grid.pack(fill='x', pady=2)

        rows = [
            ("Mw", self.mw, "(4.0 - 7.8)", 10),
            ("RJB", self.rjb, "(0.1 - 200 km)", 10),
            ("VS30", self.vs30, "(131 - 1380 m/s)", 10),
            ("FD", self.fd, "(0 - 35 km)", 10),
        ]

        for i, (label, var, range_text, width) in enumerate(rows):
            row_frame = tk.Frame(input_grid, bg='#f0f2f5')
            row_frame.grid(row=i, column=0, columnspan=3, sticky='w', padx=1, pady=1)

            tk.Label(row_frame, text=label, font=(self.font_family, 12, 'bold'),
                     bg='#f0f2f5', width=7, anchor='w').pack(side='left')
            tk.Entry(row_frame, textvariable=var, width=width, font=(self.font_family, 12),
                     bd=2, relief='groove').pack(side='left', padx=1)
            tk.Label(row_frame, text=range_text, font=(self.font_family, 9),
                     bg='#f0f2f5', fg='#555').pack(side='left', padx=2)

        row_idx = len(rows)
        row_frame = tk.Frame(input_grid, bg='#f0f2f5')
        row_frame.grid(row=row_idx, column=0, columnspan=3, sticky='w', padx=1, pady=1)

        tk.Label(row_frame, text="FM", font=(self.font_family, 12, 'bold'),
                 bg='#f0f2f5', width=7, anchor='w').pack(side='left')

        cb = ttk.Combobox(row_frame, textvariable=self.fm_str, width=10, font=(self.font_family, 12))
        cb['values'] = ['Normal', 'Reverse', 'Strike Slip']
        cb.current(1)
        cb.pack(side='left', padx=1)

        tk.Button(row_frame, text="RUN", command=self.run_prediction,
                  font=(self.font_family, 13, 'bold'), bg='#2196F3', fg='white',
                  width=10, relief='raised', bd=2, cursor='hand2'
                  ).pack(side='left', padx=5)

        out_frame = tk.LabelFrame(top_frame, text=" OUTPUTS ",
                                  font=(self.font_family, 12, 'bold'),
                                  bg='#f0f2f5', fg='#2c3e50', padx=10, pady=8)
        out_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        out_grid = tk.Frame(out_frame, bg='#f0f2f5')
        out_grid.pack(fill='both', expand=True, pady=2)

        left_col = tk.Frame(out_grid, bg='#f0f2f5')
        left_col.pack(side='left', fill='both', expand=True, padx=1)

        right_col = tk.Frame(out_grid, bg='#f0f2f5')
        right_col.pack(side='right', fill='both', expand=True, padx=1)

        left_outputs = [
            ("PGA (cm/s²)", self.pga),
            ("Ia (cm/s)", self.ia),
            ("D5-95 (s)", self.d595),
            ("CAV (cm/s)", self.cav),
        ]

        for _, (label, var) in enumerate(left_outputs):
            f = tk.Frame(left_col, bg='#f0f2f5')
            f.pack(fill='x', pady=1)
            tk.Label(f, text=label, font=(self.font_family, 12, 'bold'),
                     bg='#f0f2f5', width=12, anchor='w').pack(side='left', padx=0)
            tk.Entry(f, textvariable=var, width=9, font=(self.font_family, 12), bd=2,
                     relief='groove', state='readonly', readonlybackground='#e8f0fe',
                     justify='right').pack(side='left', padx=0)

        right_outputs = [
            ("PGV (cm/s)", self.pgv),
            ("Tm (s)", self.tm),
            ("D5-75 (s)", self.d575),
        ]

        for _, (label, var) in enumerate(right_outputs):
            f = tk.Frame(right_col, bg='#f0f2f5')
            f.pack(fill='x', pady=1)
            tk.Label(f, text=label, font=(self.font_family, 12, 'bold'),
                     bg='#f0f2f5', width=12, anchor='w').pack(side='left', padx=0)
            tk.Entry(f, textvariable=var, width=9, font=(self.font_family, 12), bd=2,
                     relief='groove', state='readonly', readonlybackground='#e8f0fe',
                     justify='right').pack(side='left', padx=0)

        file_frame = tk.LabelFrame(main, text=" EXCEL PROCESSOR ",
                                   font=(self.font_family, 11, 'bold'),
                                   bg='#f0f2f5', fg='#2c3e50', padx=10, pady=6)
        file_frame.pack(fill='x', pady=(0, 8))

        f_row = tk.Frame(file_frame, bg='#f0f2f5')
        f_row.pack(fill='x', pady=3)

        tk.Label(f_row, text="Input:", font=(self.font_family, 10, 'bold'),
                 bg='#f0f2f5').pack(side='left', padx=2)
        tk.Entry(f_row, textvariable=self.input_path, width=35, font=(self.font_family, 9),
                 bd=2, relief='groove').pack(side='left', padx=5)
        tk.Button(f_row, text="Browse", command=self.browse_input,
                  font=(self.font_family, 9, 'bold'), bg='#4CAF50', fg='white',
                  width=8, cursor='hand2').pack(side='left', padx=2)

        tk.Button(f_row, text="Process Excel", command=self.process_excel,
                  font=(self.font_family, 10, 'bold'), bg='#FF9800', fg='white',
                  width=12, relief='raised', bd=2, cursor='hand2'
                  ).pack(side='left', padx=10)

        tk.Label(f_row, text="Output:", font=(self.font_family, 10, 'bold'),
                 bg='#f0f2f5').pack(side='left', padx=10)
        tk.Entry(f_row, textvariable=self.output_path, width=30, font=(self.font_family, 9),
                 bd=2, relief='groove').pack(side='left', padx=5)

        plot_frame = tk.LabelFrame(main, text=" SPECTRAL ACCELERATION vs PERIOD ",
                                   font=(self.font_family, 12, 'bold'),
                                   bg='#f0f2f5', fg='#2c3e50', padx=10, pady=10)
        plot_frame.pack(fill='both', expand=True, pady=(5, 0))

        self.fig, self.ax = plt.subplots(figsize=(11, 5.5), dpi=100)
        self.fig.patch.set_facecolor('#fafafa')
        self.fig.subplots_adjust(left=0.10, bottom=0.18, right=0.95, top=0.92)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.status = tk.Label(main, text="Ready | Enter parameters and click RUN",
                               font=(self.font_family, 9), fg='#1565C0', bg='#f0f2f5')
        self.status.pack(anchor='w', pady=(5, 0))

    def browse_input(self):
        """Browse for Excel input file."""
        fn = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if fn:
            self.input_path.set(fn)
            base_name = os.path.splitext(os.path.basename(fn))[0]
            default_out = os.path.join(os.path.dirname(fn), f"Results_{base_name}.xlsx")
            self.output_path.set(default_out)
            self.status.config(text=f"Selected: {os.path.basename(fn)}", fg='green')

    def plot_initial(self):
        """Plot initial empty graph."""
        periods = np.array([0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                            0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4])
        y_empty = np.ones(len(periods)) * 1e-4

        self.ax.clear()
        self.ax.loglog(periods, y_empty, 'b-', linewidth=0.5, alpha=0.3, label='Ready')

        self.ax.set_xlabel('T (s)', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('PSa (cm/s²)', fontsize=14, fontweight='bold')

        self.ax.set_xticks([0.03, 0.1, 1, 4])
        self.ax.set_xticklabels(['0.03', '0.1', '1', '4'], fontsize=11)
        self.ax.set_xlim(0.03, 4)
        self.ax.set_ylim(1e-4, 10)

        self.ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.8)
        self.ax.tick_params(labelsize=11)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)
        self.ax.set_facecolor('#f8f9fa')
        self.ax.legend(loc='best', frameon=False, fontsize=11)

        self.ax.text(0.998, -0.18, 'Made by Amir Banimahd',
                     transform=self.ax.transAxes, fontsize=9,
                     verticalalignment='center', horizontalalignment='right',
                     color='#888888', style='italic')

        self.canvas.draw()

    def plot_spectra(self, results):
        """Plot spectral acceleration vs period."""
        periods = np.array([0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                            0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4])

        psa_keys = ['PSa_003sec', 'PSa_005sec', 'PSa_0075sec', 'PSa_01sec',
                    'PSa_015sec', 'PSa_02sec', 'PSa_025sec', 'PSa_03sec',
                    'PSa_04sec', 'PSa_05sec', 'PSa_075sec', 'PSa_10sec',
                    'PSa_15sec', 'PSa_20sec', 'PSa_25sec', 'PSa_30sec',
                    'PSa_35sec', 'PSa_40sec']

        y_vals = np.array([results[k] for k in psa_keys])
        y_vals = np.maximum(y_vals, 1e-10)

        self.ax.clear()
        self.ax.loglog(periods, y_vals, 'b-', linewidth=2.5, label='GMM Turkiye 2026')

        self.ax.set_xlabel('T (s)', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('PSa (cm/s²)', fontsize=14, fontweight='bold')

        self.ax.set_xticks([0.03, 0.1, 1, 4])
        self.ax.set_xticklabels(['0.03', '0.1', '1', '4'], fontsize=11)
        self.ax.set_xlim(0.03, 4)

        min_val = np.min(y_vals) * 0.8 if np.min(y_vals) > 0 else 0.001
        max_val = np.max(y_vals) * 1.5
        self.ax.set_ylim(min_val, max_val)

        self.ax.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=0.8)
        self.ax.tick_params(labelsize=11)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)
        self.ax.set_facecolor('#f8f9fa')
        self.ax.legend(loc='best', frameon=False, fontsize=11)

        self.ax.text(0.998, -0.18, 'Made by Amir Banimahd',
                     transform=self.ax.transAxes, fontsize=9,
                     verticalalignment='center', horizontalalignment='right',
                     color='#888888', style='italic')

        self.canvas.draw()

    def run_prediction(self):
        """Run prediction with current inputs."""
        if self.predictor is None:
            messagebox.showerror("Error", "Predictor not loaded. Check weights file.")
            return

        try:
            mw = self.mw.get()
            rjb = self.rjb.get()
            vs30 = self.vs30.get()
            fd = self.fd.get()

            fm_map = {'Normal': 1, 'Reverse': 2, 'Strike Slip': 3}
            fm = fm_map.get(self.fm_str.get(), 2)

            if not (4 <= mw <= 7.8):
                messagebox.showerror("Error", "Mw must be between 4.0 and 7.8")
                return
            if not (0.1 <= rjb <= 200):
                messagebox.showerror("Error", "RJB must be between 0.1 and 200 km")
                return
            if not (131 <= vs30 <= 1380):
                messagebox.showerror("Error", "VS30 must be between 131 and 1380 m/s")
                return
            if not (0 <= fd <= 35):
                messagebox.showerror("Error", "FD must be between 0 and 35 km")
                return

            results = self.predictor.predict(fd, fm, mw, rjb, vs30)

            self.pga.set(f"{results['PGA']:.2f}")
            self.pgv.set(f"{results['PGV']:.2f}")
            self.ia.set(f"{results['Ia']:.3f}")
            self.d575.set(f"{results['D_5_75']:.2f}")
            self.d595.set(f"{results['D_5_95']:.2f}")
            self.tm.set(f"{results['T_m']:.2f}")
            self.cav.set(f"{results['CAV']:.2f}")

            self.plot_spectra(results)

            self.status.config(
                text=f"PGA={results['PGA']:.2f} cm/s, PGV={results['PGV']:.2f} cm/s, "
                     f"Ia={results['Ia']:.3f} cm/s, Tm={results['T_m']:.2f} s",
                fg='green'
            )

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status.config(text="Error", fg='red')

    def process_excel(self):
        """Process Excel file with multiple scenarios."""
        if self.predictor is None:
            messagebox.showerror("Error", "Predictor not loaded. Check weights file.")
            return

        inp = self.input_path.get()
        out = self.output_path.get()

        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Please select a valid input file.")
            return
        if not out:
            messagebox.showerror("Error", "Please specify an output file.")
            return

        self.status.config(text="Processing Excel...", fg='orange')

        try:
            df = pd.read_excel(inp)

            expected_cols = ['Mw', 'VS30', 'RJB', 'FD', 'FM']

            if len(df.columns) >= 5:
                for i in range(len(expected_cols)):
                    if i < len(df.columns):
                        df.columns.values[i] = expected_cols[i]
            else:
                messagebox.showerror("Error", "Excel must have at least 5 columns")
                return

            results_list = []

            for _, row in df.iterrows():
                try:
                    mw = float(row['Mw'])
                    vs30 = float(row['VS30'])
                    rjb = float(row['RJB'])
                    fd = float(row['FD'])
                    fm = int(row['FM'])

                    results = self.predictor.predict(fd, fm, mw, rjb, vs30)
                    results_list.append(results)
                except Exception:
                    results_list.append(GMMPredictor._get_default_outputs())

            output_cols = ['Mw', 'VS30', 'RJB', 'FD', 'FM',
                           'PGA', 'PGV', 'Ia', 'D5-75', 'D5-95', 'Tm', 'CAV',
                           'PSa_003sec', 'PSa_005sec', 'PSa_0075sec', 'PSa_01sec',
                           'PSa_015sec', 'PSa_02sec', 'PSa_025sec', 'PSa_03sec',
                           'PSa_04sec', 'PSa_05sec', 'PSa_075sec', 'PSa_10sec',
                           'PSa_15sec', 'PSa_20sec', 'PSa_25sec', 'PSa_30sec',
                           'PSa_35sec', 'PSa_40sec']

            out_data = []
            for i, row in df.iterrows():
                r = results_list[i] if i < len(results_list) else GMMPredictor._get_default_outputs()
                out_row = [
                    row['Mw'], row['VS30'], row['RJB'], row['FD'], row['FM'],
                    r['PGA'], r['PGV'], r['Ia'], r['D_5_75'], r['D_5_95'],
                    r['T_m'], r['CAV'],
                    r['PSa_003sec'], r['PSa_005sec'], r['PSa_0075sec'], r['PSa_01sec'],
                    r['PSa_015sec'], r['PSa_02sec'], r['PSa_025sec'], r['PSa_03sec'],
                    r['PSa_04sec'], r['PSa_05sec'], r['PSa_075sec'], r['PSa_10sec'],
                    r['PSa_15sec'], r['PSa_20sec'], r['PSa_25sec'], r['PSa_30sec'],
                    r['PSa_35sec'], r['PSa_40sec']
                ]
                out_data.append(out_row)

            out_df = pd.DataFrame(out_data, columns=output_cols)
            out_df.to_excel(out, index=False)

            self.status.config(text=f"Saved: {os.path.basename(out)}", fg='green')
            messagebox.showinfo("Success", f"Results saved to:\n{out}")

        except Exception as e:
            self.status.config(text="Error", fg='red')
            messagebox.showerror("Error", str(e))


def run_app():
    """Run the main application."""
    root = tk.Tk()
    app = GMMApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()