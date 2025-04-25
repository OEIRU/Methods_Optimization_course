import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from functools import partial
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

class OptimizationTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.f_count = 0
        self.grad_count = 0
        self.hessian_count = 0

class ParallelGradient:
    def __init__(self, f, tracker, h=1e-6):
        self.f = f
        self.tracker = tracker
        self.h = h
    def _calc_partial(self, x, i):
        x_plus = x.copy()
        x_plus[i] += self.h
        x_minus = x.copy()
        x_minus[i] -= self.h
        return (self.f(x_plus) - self.f(x_minus)) / (2 * self.h)
    def compute(self, x):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda i: self._calc_partial(x, i), 
                range(len(x))
            ))
        self.tracker.grad_count += 2 * len(x)
        return np.array(results)

def make_function_with_counter(f, tracker):
    def wrapped(x):
        tracker.f_count += 1
        return f(x)
    return wrapped

def _f_quadratic(x):
    return 100 * (x[1] - x[0])**2 + (1 - x[0])**2

def _f_rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def _f_multi_exponential(x, params):
    A1, a1, b1, c1, d1, A2, a2, b2, c2, d2 = params
    term1 = A1 * np.exp(-((x[0]-a1)/b1)**2 - ((x[1]-c1)/d1)**2)
    term2 = A2 * np.exp(-((x[0]-a2)/b2)**2 - ((x[1]-c2)/d2)**2)
    return term1 + term2

def grad(f, x, tracker, h=1e-6):
    pg = ParallelGradient(f, tracker, h)
    return pg.compute(x)

def hessian(f, x, tracker, h=1e-6):
    n = len(x)
    hess = np.zeros((n, n))
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                futures.append(executor.submit(
                    lambda: (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
                ))  
        for idx, future in enumerate(futures):
            i = idx // n
            j = idx % n
            hess[i, j] = future.result()
            tracker.hessian_count += 4
    return hess

def make_plot(x_store, f, algorithm, params=None):
    x_store = np.array(x_store)
    padding = 0.5
    x1_min, x1_max = x_store[:, 0].min()-padding, x_store[:, 0].max()+padding
    x2_min, x2_max = x_store[:, 1].min()-padding, x_store[:, 1].max()+padding
    
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    if params is not None:
        Z = f([X1, X2], params)
    else:
        Z = f([X1, X2])
    
    plt.figure(figsize=(10, 8))
    plt.title(f"{algorithm}\nFinal Point: {x_store[-1].round(4)}\nIterations: {len(x_store)}")
    contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.scatter(x_store[:, 0], x_store[:, 1], c='red', s=50, edgecolors='white')
    plt.plot(x_store[:, 0], x_store[:, 1], c='white', lw=1, alpha=0.7)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
def make_table(x_store, f_store, nabla_store, a_store, s_store, H_store=None):
    data = []
    for i in range(len(x_store)):
        row = {
            'Iteration': i + 1,
            'x': np.round(x_store[i][0], 6),
            'y': np.round(x_store[i][1], 6),
            'f(x,y)': np.round(f_store[i], 6),
            '∇f_x': np.round(nabla_store[i][0], 4),
            '∇f_y': np.round(nabla_store[i][1], 4),
            'Step_x': np.round(s_store[i][0], 6) if i < len(s_store) else '-',
            'Step_y': np.round(s_store[i][1], 6) if i < len(s_store) else '-',
            'α': np.round(a_store[i], 6) if i < len(a_store) else '-'
        }
        if H_store and i < len(H_store):
            row['H'] = str(np.round(H_store[i], 3)).replace('\n', ' ')
        else:
            row['H'] = '-'
        data.append(row)
    return pd.DataFrame(data).set_index('Iteration')

    
def line_search(f, x, p, nabla, tracker, find_max=False, c1=1e-4, c2=0.9, max_iter=100):
    alpha = 1.0
    fx = f(x)
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = f(x_new)
        grad_new = grad(f, x_new, tracker)
        if find_max:
            armijo = fx_new >= fx + c1 * alpha * np.dot(nabla, p)
            curvature = np.dot(grad_new, p) <= c2 * np.dot(nabla, p)
        else:
            armijo = fx_new <= fx + c1 * alpha * np.dot(nabla, p)
            curvature = np.dot(grad_new, p) >= c2 * np.dot(nabla, p)
        if armijo and curvature:
            return alpha
        alpha *= 0.5
    return alpha

def newton_method(f, x0, eps, tracker, plot=True, table=True, find_max=False, max_iter=5000):
    tracker.reset()
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    f_values = [f(x)]
    gradients = [grad(f, x, tracker)]
    alphas = []
    steps = []
    hessians = []
    for _ in range(max_iter):
        grad_current = gradients[-1]
        hess = hessian(f, x, tracker)
        hessians.append(hess)
        try:
            p = np.linalg.solve(hess, -grad_current)
            if find_max:
                p = -p
            direction_ok = np.dot(p, grad_current) < -1e-15 if not find_max else np.dot(p, grad_current) > 1e-8
            if not direction_ok:
                p = grad_current if find_max else -grad_current
        except np.linalg.LinAlgError:
            p = grad_current if find_max else -grad_current
        alpha = line_search(f, x, p, grad_current, tracker, find_max)
        s = alpha * p
        x_new = x + s
        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad(f, x_new, tracker))
        alphas.append(alpha)
        steps.append(s)
        grad_norm = np.linalg.norm(gradients[-1])
        step_norm = np.linalg.norm(s)
        if grad_norm < eps or step_norm < 1e-8:
            break
        x = x_new

    # Формирование нового вывода
    output_lines = [
        "Newton Method",
        f"Number of iterations objective function: {tracker.f_count}",
        f"Number of iterations: {len(trajectory)}",
        f"Calculation accuracy: {eps:.3f}",
        " i (x, y) f(x, y) S lambda angle delta(X) delta(Y) delta(f) Gradient"
    ]
    for i in range(len(trajectory)):
        x_prev = trajectory[i-1] if i > 0 else trajectory[i]
        x_curr = trajectory[i]
        f_prev = f_values[i-1] if i > 0 else f_values[i]
        f_curr = f_values[i]
        grad_curr = gradients[i]
        alpha = alphas[i-1] if i > 0 else 0
        s = steps[i-1] if i > 0 else np.zeros_like(x_curr)
        delta_x = x_curr[0] - x_prev[0]
        delta_y = x_curr[1] - x_prev[1]
        delta_f = f_curr - f_prev
        angle = np.arctan2(delta_y, delta_x) if (delta_x != 0 or delta_y != 0) else 0
        output_lines.append(
            f"{i} ({x_curr[0]:.6f}, {x_curr[1]:.6f}) {f_curr:.6f} {s} {alpha:.6f} "
            f"{angle:.6f} {delta_x:.6f} {delta_y:.6f} {delta_f:.6f} {grad_curr}"
        )

    output_text = "\n".join(output_lines)
    print(output_text)

    if plot:
        title = 'Newton Method (Maximization)' if find_max else 'Newton Method'
        make_plot(np.array(trajectory), f, title)
    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)
    return trajectory[-1]

def dfp_method(f, x0, eps, tracker, plot=True, table=True, find_max=False, max_iter=500):
    tracker.reset()
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    grad_current = grad(f, x, tracker)
    trajectory = [x.copy()]
    f_values = [f(x)]
    gradients = [grad_current.copy()]
    alphas = []
    steps = []
    hessians = [H.copy()]
    for _ in range(max_iter):
        if find_max:
            p = H @ grad_current
            if np.dot(p, grad_current) <= 0:
                p = grad_current
                H = np.eye(n)
        else:
            p = -H @ grad_current
            if np.dot(p, grad_current) >= 0:
                p = -grad_current
                H = np.eye(n)
        if (find_max and np.dot(p, grad_current) <= 0) or (not find_max and np.dot(p, grad_current) >= 0):
            p = grad_current if find_max else -grad_current
            H = np.eye(n)
        alpha = line_search(f, x, p, grad_current, tracker, find_max=find_max)
        s = alpha * p
        x_new = x + s
        grad_new = grad(f, x_new, tracker)
        y = grad_new - grad_current
        s_norm = np.linalg.norm(s)
        y_norm = np.linalg.norm(y)
        if s_norm > 1e-8 and y_norm > 1e-8:
            s = s.reshape(-1, 1)
            y = y.reshape(-1, 1)
            rho = 1.0 / (y.T @ s + 1e-10)
            if np.abs(rho) < 1e10:
                Hy = H @ y
                H = H - (Hy @ Hy.T) / (y.T @ Hy + 1e-10) + rho * (s @ s.T)
        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad_new.copy())
        alphas.append(alpha)
        steps.append(s.flatten())
        hessians.append(H.copy())
        grad_norm = np.linalg.norm(grad_new)
        step_norm = np.linalg.norm(s)
        f_change = abs(f_values[-2] - f_values[-1])
        if grad_norm < eps or step_norm < 1e-8 or f_change < 1e-12:
            break
        x = x_new
        grad_current = grad_new

    # Формирование нового вывода
    output_lines = [
        "DFP Method",
        f"Number of iterations objective function: {tracker.f_count}",
        f"Number of iterations: {len(trajectory)}",
        f"Calculation accuracy: {eps:.3f}",
        " i (x, y) f(x, y) S lambda angle delta(X) delta(Y) delta(f) Gradient"
    ]
    for i in range(len(trajectory)):
        x_prev = trajectory[i-1] if i > 0 else trajectory[i]
        x_curr = trajectory[i]
        f_prev = f_values[i-1] if i > 0 else f_values[i]
        f_curr = f_values[i]
        grad_curr = gradients[i]
        alpha = alphas[i-1] if i > 0 else 0
        s = steps[i-1] if i > 0 else np.zeros_like(x_curr)
        delta_x = x_curr[0] - x_prev[0]
        delta_y = x_curr[1] - x_prev[1]
        delta_f = f_curr - f_prev
        angle = np.arctan2(delta_y, delta_x) if (delta_x != 0 or delta_y != 0) else 0
        output_lines.append(
            f"{i} ({x_curr[0]:.6f}, {x_curr[1]:.6f}) {f_curr:.6f} {s} {alpha:.6f} "
            f"{angle:.6f} {delta_x:.6f} {delta_y:.6f} {delta_f:.6f} {grad_curr}"
        )

    output_text = "\n".join(output_lines)
    print(output_text)

    if plot:
        make_plot(np.array(trajectory), f, 'DFP Method')
    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)
    return trajectory[-1]

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization Methods Comparison")
        self.tracker = OptimizationTracker()
        self.params = [3, 1, 2, 1, 1, 2, 3, 1, 2, 1]
        self.functions = [
            ("Quadratic Function", partial(_f_quadratic), False),
            ("Rosenbrock Function", partial(_f_rosenbrock), False),
            ("Multi-Exponential Function", 
             partial(_f_multi_exponential, params=self.params), True)
        ]
        self.create_widgets()
    
    def create_widgets(self):
        control_frame = ttk.LabelFrame(self.root, text="Parameters")
        control_frame.pack(padx=10, pady=10, fill=tk.X)
        
        # Function selection
        ttk.Label(control_frame, text="Function:").grid(row=0, column=0, padx=5, pady=5)
        self.func_var = tk.StringVar()
        self.func_combobox = ttk.Combobox(control_frame, textvariable=self.func_var,
                                        values=[f[0] for f in self.functions])
        self.func_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.func_combobox.current(0)
        
        # Initial point selection
        ttk.Label(control_frame, text="Initial Point:").grid(row=1, column=0, padx=5, pady=5)
        self.point_var = tk.StringVar()
        self.point_combobox = ttk.Combobox(control_frame, textvariable=self.point_var,
                                         values=["[-1.2, 1]", "[0, 0]", "[2, 2]", "Custom"])
        self.point_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.point_combobox.current(0)
        
        # Custom point inputs
        self.custom_point_frame = ttk.Frame(control_frame)
        ttk.Label(self.custom_point_frame, text="x:").pack(side=tk.LEFT)
        self.x_entry = ttk.Entry(self.custom_point_frame, width=5)
        self.x_entry.pack(side=tk.LEFT)
        ttk.Label(self.custom_point_frame, text="y:").pack(side=tk.LEFT)
        self.y_entry = ttk.Entry(self.custom_point_frame, width=5)
        self.y_entry.pack(side=tk.LEFT)
        
        # Tolerance selection
        ttk.Label(control_frame, text="Tolerance:").grid(row=2, column=0, padx=5, pady=5)
        self.eps_var = tk.StringVar()
        self.eps_combobox = ttk.Combobox(control_frame, textvariable=self.eps_var,
                                       values=[f"1e-{i}" for i in range(1, 8)] + ["Custom"])
        self.eps_combobox.grid(row=2, column=1, padx=5, pady=5)
        self.eps_combobox.current(0)
        
        # Run buttons
        self.run_frame = ttk.Frame(control_frame)
        self.run_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(self.run_frame, text="Run Newton Method",
                 command=lambda: self.run_optimization('newton')).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.run_frame, text="Run DFP Method",
                 command=lambda: self.run_optimization('dfp')).pack(side=tk.LEFT, padx=5)
        
        # Results area
        self.result_frame = ttk.LabelFrame(self.root, text="Results")
        self.result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Text output
        self.output_text = tk.Text(self.result_frame, height=15)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output_scroll = ttk.Scrollbar(self.result_frame, command=self.output_text.yview)
        self.output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=self.output_scroll.set)
        
        # Plot canvas
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Bindings
        self.point_combobox.bind("<<ComboboxSelected>>", self.toggle_custom_point)
        self.toggle_custom_point()

    def toggle_custom_point(self, event=None):
        if self.point_combobox.get() == "Custom":
            self.custom_point_frame.grid(row=1, column=2, padx=5, pady=5)
        else:
            self.custom_point_frame.grid_remove()

    def get_parameters(self):
        func_idx = self.func_combobox.current()
        f_name, f, find_max = self.functions[func_idx]
        f = make_function_with_counter(f, self.tracker)
        
        if self.point_combobox.get() == "Custom":
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            x0 = [x, y]
        else:
            x0 = eval(self.point_combobox.get())
        
        if self.eps_combobox.get() == "Custom":
            eps = float(self.eps_var.get())
        else:
            eps = float(self.eps_combobox.get().replace("1e-", "1e-"))
            
        return f_name, f, find_max, x0, eps

    def run_optimization(self, method):
        self.output_text.delete(1.0, tk.END)
        self.figure.clear()
        
        try:
            f_name, f, find_max, x0, eps = self.get_parameters()
        except Exception as e:
            self.output_text.insert(tk.END, f"Input Error: {str(e)}")
            return
            
        self.output_text.insert(tk.END, f"Running {method} method...\n")
        self.root.update()
        
        def optimization_thread():
            try:
                if method == 'newton':
                    result = newton_method(f, x0, eps, self.tracker, plot=False, table=True, find_max=find_max)
                else:
                    result = dfp_method(f, x0, eps, self.tracker, plot=False, table=True, find_max=find_max)
                
                # Update plot
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                x_store = np.array([[row[1]['x'], row[1]['y']] for row in result.iterrows()]) 
                
                if "Multi-Exponential" in f_name:
                    plot_function = partial(_f_multi_exponential, params=self.params)
                else:
                    plot_function = f
                
                X1, X2 = np.meshgrid(
                    np.linspace(x_store[:,0].min()-0.5, x_store[:,0].max()+0.5, 100),
                    np.linspace(x_store[:,1].min()-0.5, x_store[:,1].max()+0.5, 100))
                Z = plot_function([X1, X2])
                
                ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
                ax.plot(x_store[:,0], x_store[:,1], 'w-', lw=1)
                ax.scatter(x_store[:,0], x_store[:,1], c='red', edgecolors='white')
                ax.set_title(f"{method} Method\nFinal Point: {x_store[-1].round(4)}\nIterations: {len(x_store)}")
                self.canvas.draw()
                
                # Update text output
                self.output_text.insert(tk.END, f"\nResults for {f_name}:\n")
                self.output_text.insert(tk.END, f"Iterations: {len(result)}\n")
                self.output_text.insert(tk.END, f"Function calls: {self.tracker.f_count}\n")
                self.output_text.insert(tk.END, f"Gradient calls: {self.tracker.grad_count}\n")
                self.output_text.insert(tk.END, f"Hessian calls: {self.tracker.hessian_count}\n")
                self.output_text.insert(tk.END, f"Final point: {x_store[-1].round(6)}\n")
                self.output_text.insert(tk.END, f"Final value: {result.iloc[-1]['f(x,y)'].round(6)}\n")
                
            except Exception as e:
                self.output_text.insert(tk.END, f"Error: {str(e)}")
            finally:
                self.tracker.reset()
        
        Thread(target=optimization_thread).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()