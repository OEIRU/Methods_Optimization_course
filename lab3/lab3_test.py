"""
Модуль реализации методов штрафов и барьеров для оптимизации функций с ограничениями
"""

# Импорт необходимых библиотек
import tkinter as tk  # Для создания графического интерфейса
from tkinter import ttk, messagebox  # Виджеты Tkinter
import numpy as np  # Для численных вычислений
import matplotlib.pyplot as plt  # Для построения графиков
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Интеграция matplotlib с Tkinter
from mpl_toolkits.mplot3d import Axes3D  # 3D-графики
from scipy.optimize import minimize  # Оптимизация функций
import threading  # Многопоточность
import queue  # Очереди для межпоточного обмена данными
from typing import Tuple, List, Dict, Any, Optional, Callable  # Типизация
from enum import Enum  # Перечисления

# === Константы и перечисления ===
class ProblemType(str, Enum):
    """Типы задач оптимизации"""
    PROBLEM_A = 'a'  # x + y ≤ -1
    PROBLEM_B = 'b'  # y = x + 1

class PenaltyType(str, Enum):
    """Типы штрафных функций"""
    QUADRATIC = 'quadratic'  # Квадратичный штраф
    ABSOLUTE = 'absolute'  # Абсолютный штраф

class BarrierType(str, Enum):
    """Типы барьерных функций"""
    LOGARITHMIC = 'log'  # Логарифмический барьер
    INVERSE = 'inverse'  # Обратный барьер

class MuStrategy(str, Enum):
    """Стратегии изменения параметра μ"""
    MULTIPLY_10 = 'multiply_10'  # Увеличение в 10 раз
    ADD_5 = 'add_5'  # Добавление 5
    DIVIDE_10 = 'divide_10'  # Деление на 10
    SUBTRACT_0_5 = 'subtract_0_5'  # Вычитание 0.5

# === Целевые функции и ограничения ===
def objective_function(x: np.ndarray) -> float:
    """Целевая функция для оптимизации: f(x,y) = 4(xy)^2 + 3(x-1)^2"""
    return 4 * (x[0] * x[1])**2 + 3 * (x[0] - 1)**2

def constraint_a(x: np.ndarray) -> float:
    """Ограничение для задачи A: x + y ≤ -1"""
    return x[0] + x[1] + 1

def constraint_b(x: np.ndarray) -> float:
    """Ограничение для задачи B: y = x + 1"""
    return x[1] - x[0] - 1

# === Штрафные и барьерные функции ===
def calculate_penalty(x: np.ndarray, problem: ProblemType, 
                     penalty_type: PenaltyType, mu: float) -> float:
    """
    Вычисляет штрафную функцию в зависимости от типа задачи и метода
    Args:
        x: Точка в пространстве (x, y)
        problem: Тип задачи (A или B)
        penalty_type: Тип штрафа (квадратичный или абсолютный)
        mu: Параметр штрафа
    Returns:
        Значение штрафной функции
    """
    if problem == ProblemType.PROBLEM_A:
        constraint_value = constraint_a(x)
        if penalty_type == PenaltyType.QUADRATIC:
            return mu * max(0, constraint_value)**2  # Квадратичный штраф для задачи A
        return mu * max(0, constraint_value)  # Абсолютный штраф для задачи A
    elif problem == ProblemType.PROBLEM_B:
        constraint_value = constraint_b(x)
        if penalty_type == PenaltyType.QUADRATIC:
            return mu * constraint_value**2  # Квадратичный штраф для задачи B
        return mu * abs(constraint_value)  # Абсолютный штраф для задачи B
    return 0

def calculate_barrier(x: np.ndarray, problem: ProblemType, 
                     barrier_type: BarrierType, mu: float) -> float:
    """
    Вычисляет барьерную функцию для задачи A
    Args:
        x: Точка в пространстве (x, y)
        problem: Тип задачи (только A для барьерных методов)
        barrier_type: Тип барьера (логарифмический или обратный)
        mu: Параметр барьера
    Returns:
        Значение барьерной функции
    Raises:
        ValueError: Если используется задача B с барьерным методом
    """
    if problem == ProblemType.PROBLEM_A:
        constraint_value = -(x[0] + x[1] + 1)  # Преобразование ограничения x + y < -1
        if constraint_value <= 1e-10:  # Проверка на выход за границу области
            return float('inf')
        if barrier_type == BarrierType.LOGARITHMIC:
            return -mu * np.log(constraint_value)  # Логарифмический барьер
        elif barrier_type == BarrierType.INVERSE:
            return mu / constraint_value  # Обратный барьер
    else:
        raise ValueError("Барьерный метод не поддерживается для задачи B")
    return 0

# === Методы оптимизации ===
def solve_penalty_method(problem: ProblemType, penalty_type: PenaltyType, 
                        initial_point: List[float], mu_initial: float, 
                        mu_strategy: MuStrategy, epsilon: float) -> List[Tuple[int, np.ndarray, float]]:
    """
    Решает задачу оптимизации с использованием штрафного метода
    Args:
        problem: Тип задачи (A или B)
        penalty_type: Тип штрафной функции
        initial_point: Начальная точка оптимизации
        mu_initial: Начальное значение параметра μ
        mu_strategy: Стратегия обновления μ
        epsilon: Точность остановки
    Returns:
        Список результатов по итерациям (номер итерации, точка, значение функции)
    """
    x = np.array(initial_point, dtype=np.float64)
    mu = mu_initial
    results = []
    for iteration in range(100):
        def objective(x):
            base = objective_function(x)
            return base + calculate_penalty(x, problem, penalty_type, mu)
        res = minimize(objective, x, method='Nelder-Mead', tol=epsilon)
        x = res.x
        results.append((iteration+1, x.copy(), res.fun))
        # Обновление параметра mu по заданной стратегии
        if mu_strategy == MuStrategy.MULTIPLY_10:
            mu *= 10
        elif mu_strategy == MuStrategy.ADD_5:
            mu += 5
        # Проверка сходимости по изменению целевой функции
        if len(results) > 1 and abs(results[-1][2] - results[-2][2]) < epsilon:
            break
    return results

def solve_barrier_method(problem: ProblemType, barrier_type: BarrierType,
                        initial_point: List[float], mu_initial: float,
                        mu_strategy: MuStrategy, epsilon: float) -> List[Tuple[int, np.ndarray, float]]:
    """
    Решает задачу оптимизации с использованием барьерного метода
    Args:
        problem: Тип задачи (только A для барьерных методов)
        barrier_type: Тип барьерной функции
        initial_point: Начальная точка оптимизации
        mu_initial: Начальное значение параметра μ
        mu_strategy: Стратегия обновления μ
        epsilon: Точность остановки
    Returns:
        Список результатов по итерациям (номер итерации, точка, значение функции)
    Raises:
        ValueError: Если используется задача B с барьерным методом
    """
    if problem == ProblemType.PROBLEM_B:
        raise ValueError("Барьерный метод не поддерживается для задачи B")
    x = np.array(initial_point, dtype=np.float64)
    mu = mu_initial
    results = []
    for iteration in range(100):
        def objective(x):
            base = objective_function(x)
            return base + calculate_barrier(x, problem, barrier_type, mu)
        try:
            res = minimize(objective, x, method='Nelder-Mead', tol=epsilon)
            if not res.success:
                break
            x = res.x
            results.append((iteration+1, x.copy(), res.fun))
            # Проверка сходимости по изменению целевой функции
            if len(results) > 1 and abs(results[-1][2] - results[-2][2]) < epsilon:
                break
            # Обновление параметра μ по заданной стратегии
            if mu_strategy == MuStrategy.DIVIDE_10:
                mu /= 10
            elif mu_strategy == MuStrategy.SUBTRACT_0_5:
                mu = max(mu - 0.5, 1e-10)
        except Exception as e:
            print(f"Ошибка в барьерном методе: {e}")
            break
    return results

# === Класс для выполнения оптимизации в потоке ===
class OptimizationThread(threading.Thread):
    """Поток для выполнения оптимизации без блокировки интерфейса"""
    def __init__(self, task_func: Callable, args: Tuple):
        super().__init__()
        self.task_func = task_func  # Функция оптимизации
        self.args = args  # Аргументы для функции
        self.result_queue = queue.Queue()  # Очередь для результата

    def run(self):
        try:
            result = self.task_func(*self.args)
            self.result_queue.put(result)
        except Exception as e:
            self.result_queue.put(e)

# === GUI ===
class OptimizationApp:
    """Основное приложение для оптимизации с графическим интерфейсом"""
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Метод штрафов/барьеров")
        self.root.geometry("1200x800")
        # Настройка стилей Tkinter
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        # Создание вкладок для задач A и B
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.create_task_tab("Задача A (x+y ≤ -1)", ProblemType.PROBLEM_A)
        self.create_task_tab("Задача B (y = x+1)", ProblemType.PROBLEM_B)

    def create_task_tab(self, title: str, problem: ProblemType):
        """Создает вкладку с настройками для конкретной задачи"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        # Хранение переменных интерфейса для каждой задачи
        problem_vars = {
            'frame': frame,
            'method_var': tk.StringVar(value="penalty"),
            'penalty_type': tk.StringVar(value=PenaltyType.QUADRATIC),
            'barrier_type': tk.StringVar(value=BarrierType.LOGARITHMIC),
        }
        # Левая панель с настройками
        left_frame = ttk.LabelFrame(frame, text="Настройки")
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        ttk.Label(left_frame, text="Метод:").pack(anchor='w')
        # Радиокнопки для выбора метода
        ttk.Radiobutton(left_frame, text="Штрафные функции", 
                       variable=problem_vars['method_var'], value="penalty",
                       command=lambda: self.update_mu_strategies(problem)).pack(anchor='w')
        ttk.Radiobutton(left_frame, text="Барьерные функции", 
                       variable=problem_vars['method_var'], value="barrier",
                       command=lambda: self.update_mu_strategies(problem)).pack(anchor='w')
        # Выбор типа функции
        ttk.Label(left_frame, text="Тип функции:").pack(anchor='w')
        penalty_menu = ttk.OptionMenu(left_frame, problem_vars['penalty_type'], 
                                    PenaltyType.QUADRATIC, *PenaltyType)
        barrier_menu = ttk.OptionMenu(left_frame, problem_vars['barrier_type'], 
                                     BarrierType.LOGARITHMIC, *BarrierType)
        penalty_menu.pack(anchor='w')
        barrier_menu.pack(anchor='w')
        problem_vars['penalty_menu'] = penalty_menu
        problem_vars['barrier_menu'] = barrier_menu
        # Поля ввода параметров
        problem_vars['x0'] = self.create_input_field(left_frame, "x:", "0.0")
        problem_vars['y0'] = self.create_input_field(left_frame, "y:", "0.0")
        problem_vars['mu_initial'] = self.create_input_field(left_frame, "Начальный μ:", "1.0")
        problem_vars['epsilon'] = self.create_input_field(left_frame, "Точность ε:", "0.001")
        # Выбор стратегии изменения μ
        ttk.Label(left_frame, text="Стратегия μ:").pack(anchor='w')
        mu_strategy = ttk.Combobox(left_frame, values=[s.value for s in MuStrategy])
        mu_strategy.set(MuStrategy.MULTIPLY_10)
        mu_strategy.pack(anchor='w')
        problem_vars['mu_strategy'] = mu_strategy
        # Кнопка запуска оптимизации
        solve_button = ttk.Button(left_frame, text="Решить", 
                                 command=lambda: self.run_optimization(problem))
        solve_button.pack(pady=10)
        problem_vars['solve_button'] = solve_button
        # Правая панель с результатами и графиками
        right_frame = ttk.Frame(frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        # Текстовое поле для вывода результатов
        result_text = tk.Text(right_frame, height=10, width=50, font=("Courier", 10))
        result_text.pack(pady=5)
        problem_vars['result_text'] = result_text
        # График сходимости
        fig, ax = plt.subplots(figsize=(5, 3))
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(pady=5)
        problem_vars['ax'] = ax
        problem_vars['canvas'] = canvas
        # 3D-график целевой функции
        fig_3d = plt.figure(figsize=(6, 5))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        canvas_3d = FigureCanvasTkAgg(fig_3d, master=right_frame)
        canvas_3d.get_tk_widget().pack(pady=5)
        problem_vars['ax_3d'] = ax_3d
        problem_vars['canvas_3d'] = canvas_3d
        # Панель управления графиком
        problem_vars['toolbar'] = NavigationToolbar2Tk(canvas_3d, right_frame)
        problem_vars['toolbar'].update()
        setattr(self, f"{problem}_vars", problem_vars)
        self.update_mu_strategies(problem)

    def create_input_field(self, parent: ttk.Frame, label: str, default_value: str) -> ttk.Entry:
        """Создает поле ввода с подписью"""
        frame = ttk.Frame(parent)
        frame.pack(anchor='w', pady=2)
        ttk.Label(frame, text=label).pack(side='left')
        entry = ttk.Entry(frame, width=10)
        entry.insert(0, default_value)
        entry.pack(side='left', padx=5)
        return entry

    def update_mu_strategies(self, problem: ProblemType):
        """Обновляет доступные стратегии изменения μ в зависимости от выбранного метода"""
        problem_vars = getattr(self, f"{problem}_vars")
        method = problem_vars['method_var'].get()
        mu_strategy = problem_vars['mu_strategy']
        # Для штрафных методов доступны стратегии увеличения μ
        if method == "penalty":
            mu_strategy['values'] = [s.value for s in MuStrategy if 'MULTIPLY' in s.name or 'ADD' in s.name]
            mu_strategy.set(MuStrategy.MULTIPLY_10)
            problem_vars['penalty_menu'].pack()
            problem_vars['barrier_menu'].pack_forget()
        else:
            # Для барьерных методов только уменьшение μ
            if problem == ProblemType.PROBLEM_B:
                problem_vars['method_var'].set("penalty")
                messagebox.showerror("Ошибка", "Для задачи B доступен только метод штрафов")
                self.update_mu_strategies(problem)
                return
            mu_strategy['values'] = [s.value for s in MuStrategy if 'DIVIDE' in s.name or 'SUBTRACT' in s.name]
            mu_strategy.set(MuStrategy.DIVIDE_10)
            problem_vars['penalty_menu'].pack_forget()
            problem_vars['barrier_menu'].pack()

    def run_optimization(self, problem: ProblemType):
        """Запускает процесс оптимизации в отдельном потоке"""
        problem_vars = getattr(self, f"{problem}_vars")
        problem_vars['solve_button'].config(state=tk.DISABLED)
        try:
            # Получение параметров из интерфейса
            x0 = float(problem_vars['x0'].get())
            y0 = float(problem_vars['y0'].get())
            initial_point = [x0, y0]
            mu_initial = float(problem_vars['mu_initial'].get())
            mu_strategy = problem_vars['mu_strategy'].get()
            epsilon = float(problem_vars['epsilon'].get())
            method = problem_vars['method_var'].get()
            # Проверка начальной точки для барьерного метода
            if method == "barrier" and problem == ProblemType.PROBLEM_A and (x0 + y0) >= -1:
                messagebox.showerror("Ошибка", "Начальная точка должна удовлетворять x + y < -1")
                problem_vars['solve_button'].config(state=tk.NORMAL)
                return
            # Выбор соответствующего метода оптимизации
            if method == "penalty":
                task_func = solve_penalty_method
                args = (problem, problem_vars['penalty_type'].get(), initial_point, 
                       mu_initial, mu_strategy, epsilon)
            else:
                task_func = solve_barrier_method
                args = (problem, problem_vars['barrier_type'].get(), initial_point, 
                       mu_initial, mu_strategy, epsilon)
            # Запуск оптимизации в потоке
            self.thread = OptimizationThread(task_func, args)
            self.thread.start()
            self.root.after(100, self.check_thread, problem)
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите числовые значения.")
            problem_vars['solve_button'].config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            problem_vars['solve_button'].config(state=tk.NORMAL)

    def check_thread(self, problem: ProblemType):
        """Проверяет завершение потока оптимизации"""
        problem_vars = getattr(self, f"{problem}_vars")
        if self.thread.is_alive():
            self.root.after(100, self.check_thread, problem)
        else:
            try:
                result = self.thread.result_queue.get_nowait()
                if isinstance(result, Exception):
                    raise result
                self.display_results(result, problem)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
            finally:
                problem_vars['solve_button'].config(state=tk.NORMAL)

    def display_results(self, results: List[Tuple[int, np.ndarray, float]], problem: ProblemType):
        """Отображает результаты оптимизации в интерфейсе"""
        problem_vars = getattr(self, f"{problem}_vars")
        problem_vars['result_text'].delete(1.0, tk.END)
        # Вывод результатов по итерациям
        for res in results:
            problem_vars['result_text'].insert(
                tk.END, 
                f"Iter {res[0]:2d}: x={res[1][0]:.6f}, y={res[1][1]:.6f}, f={res[2]:.6f}\n"
            )
        # График сходимости
        problem_vars['ax'].clear()
        iterations = [r[0] for r in results]
        values = [r[2] for r in results]
        problem_vars['ax'].plot(iterations, values, 'b.-')
        problem_vars['ax'].set_xlabel("Итерация")
        problem_vars['ax'].set_ylabel("Значение функции")
        problem_vars['ax'].grid(True)
        problem_vars['canvas'].draw()
        # 3D-график
        self.plot_3d_function(problem, results)

    def plot_3d_function(self, problem: ProblemType, results: List[Tuple[int, np.ndarray, float]]):
        """Строит 3D-график целевой функции и траекторию оптимизации"""
        problem_vars = getattr(self, f"{problem}_vars")
        ax_3d = problem_vars['ax_3d']
        ax_3d.clear()
        # Создание сетки для графика
        x_vals = np.linspace(-5, 5, 100)
        y_vals = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = 4 * (X * Y)**2 + 3 * (X - 1)**2
        # Построение поверхности
        ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        # Ограничения для задач
        if problem == ProblemType.PROBLEM_A:
            ax_3d.contour(X, Y, X + Y + 1, [0], colors='red', linestyles='--')
        else:
            x_line = np.linspace(-5, 5, 100)
            y_line = x_line + 1
            z_line = 4 * (x_line * y_line)**2 + 3 * (x_line - 1)**2
            ax_3d.plot(x_line, y_line, z_line, color='red', linewidth=2)
        # Траектория оптимизации
        if results:
            x_path = [r[1][0] for r in results]
            y_path = [r[1][1] for r in results]
            z_path = [4*(x*y)**2 + 3*(x-1)**2 for x, y in zip(x_path, y_path)]
            ax_3d.plot(x_path, y_path, z_path, 'b.-', markersize=3)
            x0 = float(problem_vars['x0'].get())
            y0 = float(problem_vars['y0'].get())
            z0 = 4*(x0*y0)**2 + 3*(x0-1)**2
            ax_3d.scatter(x0, y0, z0, c='g', s=50, label="Начальная точка")
        # Настройка подписей
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("f(x, y)")
        ax_3d.legend()
        problem_vars['canvas_3d'].draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()