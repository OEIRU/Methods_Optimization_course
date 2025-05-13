import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize

# Стилизация
BG_COLOR = "#f0f0f0"
BUTTON_COLOR = "#e1e1e1"
FONT = ("Arial", 10)

# Определение целевых функций и ограничений
def objective_a(x):
    return 4*(x[0]*x[1])**2 + 3*(x[0]-1)**2

def constraint_a(x):
    return x[0] + x[1] + 1  # x + y <= -1

def objective_b(x):
    return 4*(x[0]*x[1])**2 + 3*(x[0]-1)**2

def constraint_b(x):
    return x[1] - x[0] - 1  # y = x + 1

# Штрафные функции
def penalty_function(x, problem, penalty_type, mu):
    if problem == 'a':
        constraint_val = constraint_a(x)
        if penalty_type == 'quadratic':
            return mu * max(0, constraint_val)**2
        return mu * max(0, constraint_val)
    elif problem == 'b':
        constraint_val = constraint_b(x)
        if penalty_type == 'quadratic':
            return mu * constraint_val**2
        return mu * abs(constraint_val)

# Барьерные функции
def barrier_function(x, problem, barrier_type, mu):
    if problem == 'a':
        # Правильное вычисление ограничения x + y <= -1 -> -(x + y + 1) >= 0
        constraint_val = - (x[0] + x[1] + 1)
        if constraint_val <= 1e-10:  # Добавлен небольшой допуск
            return float('inf')
        try:
            if barrier_type == 'log':
                return -mu * np.log(constraint_val)
            elif barrier_type == 'inverse':
                return mu / constraint_val
        except:
            return float('inf')
    return 0

# Метод штрафных функций
def solve_penalty_method(problem, penalty_type, initial_point, mu_initial, mu_strategy, epsilon, max_iterations=100):
    x = initial_point.copy()
    mu = mu_initial
    results = []
    for i in range(max_iterations):
        def extended_objective(x):
            obj = objective_a(x) if problem == 'a' else objective_b(x)
            return obj + penalty_function(x, problem, penalty_type, mu)
        
        result = minimize(extended_objective, x, method='Nelder-Mead')
        x = result.x
        results.append((i+1, x, result.fun))
        
        if mu_strategy == 'multiply_10':
            mu *= 10
        elif mu_strategy == 'add_5':
            mu += 5
            
        if i > 0 and abs(results[-1][2] - results[-2][2]) < epsilon:
            break
    return results

# Метод барьерных функций
def solve_barrier_method(problem, barrier_type, initial_point, mu_initial, mu_strategy, epsilon, max_iterations=100):
    x = np.array(initial_point.copy(), dtype=np.float64)
    mu = mu_initial
    results = []
    
    for i in range(max_iterations):
        def extended_objective(x):
            obj = objective_a(x) if problem == 'a' else objective_b(x)
            barrier = barrier_function(x, problem, barrier_type, mu)
            return obj + barrier
        
        try:
            result = minimize(extended_objective, x, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-8})
            if not result.success:
                break
                
            x = result.x
            current_value = result.fun - barrier_function(x, problem, barrier_type, mu)
            
            results.append((i+1, x.copy(), current_value))
            
            # Проверка сходимости
            if i > 0 and abs(results[-1][2] - results[-2][2]) < epsilon:
                break
                
            # Обновление коэффициента
            if mu_strategy == 'divide_10':
                mu /= 10
            elif mu_strategy == 'subtract_0_5':
                mu = max(mu - 0.5, 1e-10)
                
        except Exception as e:
            print(f"Ошибка оптимизации: {e}")
            break
            
    return results
class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Нелінійне програмування")
        self.root.configure(bg=BG_COLOR)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Настройка стилей
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR, font=FONT)
        self.style.configure('TRadiobutton', background=BG_COLOR, font=FONT)
        self.style.configure('TButton', font=FONT, background=BUTTON_COLOR)
        self.style.configure('Header.TLabel', font=("Arial", 12, "bold"))
        
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Панель задач
        tasks_panel = ttk.Frame(main_frame)
        tasks_panel.pack(fill='both', expand=True)
        
        # Задача A
        self.task_a_frame = ttk.LabelFrame(tasks_panel, text="Задача А (x+y ≤ -1)")
        self.task_a_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Задача B
        self.task_b_frame = ttk.LabelFrame(tasks_panel, text="Задача Б (y = x+1)")
        self.task_b_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.create_task_widgets(self.task_a_frame, 'a')
        self.create_task_widgets(self.task_b_frame, 'b')
        
    def create_task_widgets(self, parent, problem):
        # Контейнеры для группировки
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Блок настроек метода
        method_frame = ttk.LabelFrame(left_frame, text="Настройки метода")
        method_frame.pack(fill='x', padx=5, pady=5)
        
        method_var = tk.StringVar(value="penalty")
        ttk.Radiobutton(method_frame, text="Штрафные функции", variable=method_var, 
                        value="penalty").pack(anchor='w')
        ttk.Radiobutton(method_frame, text="Барьерные функции", variable=method_var, 
                       value="barrier").pack(anchor='w')
        
        # Блок типа функции
        func_frame = ttk.LabelFrame(left_frame, text="Тип функции")
        func_frame.pack(fill='x', padx=5, pady=5)
        
        penalty_type = tk.StringVar(value="quadratic")
        barrier_type = tk.StringVar(value="log")
        
        ttk.OptionMenu(func_frame, penalty_type, "quadratic", 
                      "quadratic", "absolute").pack(fill='x')
        ttk.OptionMenu(func_frame, barrier_type, "log", 
                      "log", "inverse").pack(fill='x')
        
        # Блок параметров
        param_frame = ttk.LabelFrame(left_frame, text="Параметры")
        param_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(param_frame, text="Начальный коэффициент:").pack(anchor='w')
        mu_initial = ttk.Entry(param_frame)
        mu_initial.insert(0, "1.0")
        mu_initial.pack(fill='x')
        
        ttk.Label(param_frame, text="Стратегия изменения:").pack(anchor='w')
        mu_strategy = ttk.Combobox(param_frame, values=["multiply_10", "add_5", 
                                                          "divide_10", "subtract_0_5"])
        mu_strategy.set("multiply_10")
        mu_strategy.pack(fill='x')
        
        # Блок начальной точки
        point_frame = ttk.LabelFrame(left_frame, text="Начальная точка")
        point_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(point_frame, text="x:").pack(side='left')
        x0 = ttk.Entry(point_frame, width=8)
        x0.insert(0, "0.0")
        x0.pack(side='left', padx=5)
        
        ttk.Label(point_frame, text="y:").pack(side='left')
        y0 = ttk.Entry(point_frame, width=8)
        y0.insert(0, "0.0")
        y0.pack(side='left', padx=5)
        
        # Блок точности
        ttk.Label(left_frame, text="Точность ε:").pack(anchor='w')
        epsilon = ttk.Entry(left_frame)
        epsilon.insert(0, "0.001")
        epsilon.pack(fill='x', padx=5, pady=5)
        
        # Кнопка запуска
        ttk.Button(left_frame, text="Решить", 
                 command=lambda: self.run_optimization(
                     problem=problem,
                     method=method_var.get(),
                     penalty_type=penalty_type.get(),
                     barrier_type=barrier_type.get(),
                     mu_initial=mu_initial.get(),
                     mu_strategy=mu_strategy.get(),
                     x0=x0.get(),
                     y0=y0.get(),
                     epsilon=epsilon.get(),
                     result_frame=right_frame
                 )).pack(fill='x', padx=5, pady=10)
        
        # Область результатов
        results_frame = ttk.LabelFrame(right_frame, text="Результаты")
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        result_text = tk.Text(results_frame, height=10, width=40, 
                                 font=('Consolas', 9))
        result_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # График
        fig, ax = plt.subplots(figsize=(5, 3))
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Сохраняем элементы управления
        setattr(self, f"{problem}_vars", {
            'method_var': method_var,
            'penalty_type': penalty_type,
            'barrier_type': barrier_type,
            'mu_initial': mu_initial,
            'mu_strategy': mu_strategy,
            'x0': x0,
            'y0': y0,
            'epsilon': epsilon,
            'result_text': result_text,
            'fig': fig,
            'ax': ax,
            'canvas': canvas
        })
        
    def run_optimization(self, problem, method, penalty_type, barrier_type, 
                        mu_initial, mu_strategy, x0, y0, epsilon, result_frame):
        try:
            params = {
                'problem': problem,
                'initial_point': [float(x0), float(y0)],
                'mu_initial': float(mu_initial),
                'mu_strategy': mu_strategy,
                'epsilon': float(epsilon)
            }
            
            if method == "penalty":
                params['penalty_type'] = penalty_type
                results = solve_penalty_method(**params)
            else:
                if problem == 'b':
                    messagebox.showerror("Ошибка", "Для задачи б доступен только метод штрафов")
                    return
                if sum(params['initial_point']) >= -1 and problem == 'a':
                    messagebox.showerror("Ошибка", "Начальная точка должна удовлетворять x + y < -1")
                    return
                params['barrier_type'] = barrier_type
                results = solve_barrier_method(**params)
            
            self.display_results(results, problem)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        
    def display_results(self, results, problem):
        elements = getattr(self, f"{problem}_vars")
        elements['result_text'].delete(1.0, tk.END)
        for res in results:
            elements['result_text'].insert(tk.END, 
                f"Iter {res[0]:2d}: x={res[1][0]:.6f}\n"
                f"       y={res[1][1]:.6f}\n"
                f"       f={res[2]:.6f}\n{'-'*30}\n")
        
        elements['ax'].clear()
        iterations = [r[0] for r in results]
        values = [r[2] for r in results]
        elements['ax'].plot(iterations, values, 'b.-', linewidth=1)
        elements['ax'].set_xlabel("Итерация", fontsize=9)
        elements['ax'].set_ylabel("Значение целевой функции", fontsize=9)
        elements['ax'].set_title(f"График сходимости ({problem.upper()})", fontsize=10)
        elements['ax'].grid(True, linestyle='--', alpha=0.6)
        elements['canvas'].draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()