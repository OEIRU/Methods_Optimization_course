import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from functools import partial
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Dict, Optional
import logging
import traceback

# Настройка логирования с временной меткой и уровнем INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Класс для отслеживания количества вызовов функций оптимизации
class OptimizationTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс всех счетчиков"""
        self.f_count = 0      # Кол-во вызовов целевой функции
        self.grad_count = 0   # Кол-во вызовов градиента
        self.hessian_count = 0  # Кол-во вызовов гессиана

# Параллельное вычисление градиентов с использованием потоков
class ParallelGradient:
    def __init__(self, f: Callable, tracker: OptimizationTracker, h: float = 1e-6):
        self.f = f
        self.tracker = tracker
        self.h = h  # Шаг для конечных разностей
    
    def _calc_partial(self, x: np.ndarray, i: int) -> float:
        """Вычисление частной производной по i-му направлению"""
        x_plus = x.copy()
        x_plus[i] += self.h
        x_minus = x.copy()
        x_minus[i] -= self.h
        result = (self.f(x_plus) - self.f(x_minus)) / (2 * self.h)
        return result
    
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Параллельное вычисление градиента"""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda i: self._calc_partial(x, i), range(len(x))))
        self.tracker.grad_count += 2 * len(x)  # Два вызова функции на компоненту
        return np.array(results)

# Обертка для подсчета вызовов целевой функции
def make_function_with_counter(f: Callable, tracker: OptimizationTracker) -> Callable:
    def wrapped(x: np.ndarray) -> float:
        tracker.f_count += 1
        return f(x)
    return wrapped

# Определение тестовых функций и их производных
def _f_quadratic(x: np.ndarray) -> float:
    """Квадратичная функция: 100*(x1-x2)^2 + (1-x1)^2"""
    return 100 * (x[1] - x[0])**2 + (1 - x[0])**2

def grad_quadratic(x: np.ndarray) -> np.ndarray:
    """Аналитический градиент квадратичной функции"""
    return np.array([
        -200 * (x[1] - x[0]) - 2 * (1 - x[0]),  # ∂f/∂x1
        200 * (x[1] - x[0])                     # ∂f/∂x2
    ])

def hessian_quadratic(x: np.ndarray) -> np.ndarray:
    """Аналитический гессиан квадратичной функции"""
    return np.array([
        [202, -200],
        [-200, 200]
    ], dtype=np.float64)

def _f_rosenbrock(x: np.ndarray) -> float:
    """Функция Розенброка: 100*(x2-x1^2)^2 + (1-x1)^2"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Аналитический градиент функции Розенброка"""
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 
                    200 * (x[1] - x[0]**2)])

def hessian_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Аналитический гессиан функции Розенброка"""
    return np.array([
        [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ], dtype=np.float64)

def _f_multi_exponential(x: np.ndarray, params: List[float]) -> float:
    """Многокомпонентная экспоненциальная функция с 2 пиками"""
    A1, a1, b1, c1, d1, A2, a2, b2, c2, d2 = params
    term1 = A1 * np.exp(-((x[0]-a1)/b1)**2 - ((x[1]-c1)/d1)**2)
    term2 = A2 * np.exp(-((x[0]-a2)/b2)**2 - ((x[1]-c2)/d2)**2)
    return term1 + term2

# Численное вычисление градиентов и гессианов
def grad(f: Callable, x: np.ndarray, tracker: OptimizationTracker, h: float = 1e-6) -> np.ndarray:
    """Численное вычисление градиента с параллелизацией"""
    pg = ParallelGradient(f, tracker, h)
    return pg.compute(x)

def hessian(x: np.ndarray, f: Callable, tracker: OptimizationTracker, h: float = 1e-6) -> np.ndarray:
    """Численное вычисление гессиана с использованием конечных разностей"""
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    points = []
    
    # Генерация точек для вычисления вторых производных
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
            points.extend([x_pp, x_pm, x_mp, x_mm])
    
    # Параллельное вычисление значений функции
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: f(p), points))
    
    # Подсчет количества вызовов (4 вызова на каждый элемент матрицы)
    tracker.hessian_count += 4 * n * n
    
    idx = 0
    for i in range(n):
        for j in range(n):
            f_pp = results[idx]
            f_pm = results[idx+1]
            f_mp = results[idx+2]
            f_mm = results[idx+3]
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
            idx += 4
    return hess

# Визуализация траектории оптимизации
def make_plot(x_store: List[np.ndarray], f: Callable, algorithm: str, 
              params: Optional[List[float]] = None) -> None:
    """Построение контурного графика с траекторией оптимизации"""
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
    plt.title(f"{algorithm}\n"
              f"Final Point: {x_store[-1].round(4)}\n"
              f"Iterations: {len(x_store)}")
    contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.scatter(x_store[:, 0], x_store[:, 1], c='red', s=50, edgecolors='white')
    plt.plot(x_store[:, 0], x_store[:, 1], c='white', lw=1, alpha=0.7)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Создание таблицы результатов оптимизации
def make_table(x_store: List[np.ndarray], f_store: List[float], 
               nabla_store: List[np.ndarray], a_store: List[float], 
               s_store: List[np.ndarray], H_store: Optional[List[np.ndarray]] = None) -> pd.DataFrame:
    """Создание таблицы с итерациями оптимизации"""
    data = []
    for i in range(len(x_store)):
        row = {
            'Iteration': i + 1,
            'x': np.round(x_store[i][0], 6),
            'y': np.round(x_store[i][1], 6),
            'f(x,y)': np.round(f_store[i], 6),
            '∇f_x': np.round(nabla_store[i][0], 4),
            '∇f_Y': np.round(nabla_store[i][1], 4),
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

# Линейный поиск с условиями Армихо и кривизны
def line_search(f: Callable, x: np.ndarray, p: np.ndarray, nabla: np.ndarray, 
                tracker: OptimizationTracker, find_max: bool = False, c1: float = 1e-4, 
                c2: float = 0.9, max_iter: int = 100) -> float:
    """Определение оптимального шага alpha для направления p"""
    alpha = 1.0
    min_alpha = 1e-10
    fx = f(x)
    
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = f(x_new)
        grad_new = grad(f, x_new, tracker)
        
        # Условия Армихо и кривизны
        if find_max:
            armijo = fx_new >= fx + c1 * alpha * np.dot(nabla, p)
            curvature = np.dot(grad_new, p) >= c2 * np.dot(nabla, p)
        else:
            armijo = fx_new <= fx + c1 * alpha * np.dot(nabla, p)
            curvature = np.dot(grad_new, p) >= c2 * np.dot(nabla, p)
        
        if armijo and curvature or alpha < min_alpha:
            return alpha
        alpha *= 0.5
    return alpha

# Метод Ньютона с гессианом
def newton_method(f: Callable, x0: np.ndarray, eps: float, tracker: OptimizationTracker, 
                  grad_f: Callable = None, hess_f: Callable = None, plot: bool = True, 
                  table: bool = True, find_max: bool = False, max_iter: int = 5000) -> Optional[pd.DataFrame]:
    """Метод Ньютона для минимизации/максимизации функции.
    
    Использует аналитический или численный гессиан для определения направления спуска.
    Поддерживает как минимизацию, так и максимизацию через флаг `find_max`.
    """
    tracker.reset()  # Сброс счётчиков вызовов функций
    x = np.array(x0, dtype=float)  # Текущая точка
    trajectory = [x.copy()]  # История точек (траектория)
    f_values = [f(x)]  # Значения целевой функции
    
    # Выбор функции для вычисления градиента и гессиана
    # Приоритет аналитическим производным, если они доступны
    grad_func = grad_f if grad_f else ParallelGradient(f, tracker).compute
    hess_func = hess_f if hess_f else partial(hessian, f=f, tracker=tracker)
    
    gradients = [grad_func(x)]  # Градиенты на каждой итерации
    alphas = []  # Шаги линейного поиска
    steps = []  # Векторы шагов
    hessians = []  # Хранение матриц гессиана
    
    for _ in range(max_iter):
        grad_current = gradients[-1]  # Текущий градиент
        
        try:
            hess = hess_func(x)  # Вычисление гессиана
            hessians.append(hess.copy())
            
            # Проверка положительной определённости гессиана
            # Для корректности метода Ньютона матрица должна быть положительно определена
            eigenvalues = np.linalg.eigvalsh(hess)
            min_eigen = np.min(eigenvalues)
            if min_eigen < 1e-10:  # Если есть близкие к нулю собственные значения
                hess = (hess + hess.T) / 2  # Симметризация матрицы
                # hess += (1e-10 - min_eigen) * np.eye(len(x), dtype=float)  # Альтернативное регуляризирование
            
            # Определение направления поиска
            # Решаем систему H * p = -grad (для минимизации)
            p = np.linalg.solve(hess, -grad_current)
            if find_max:  # Для максимизации инвертируем направление
                p = -p
            
            # Проверка направления: должно быть согласовано с градиентом
            direction_ok = np.dot(p, grad_current) > 1e-8 if find_max else np.dot(p, grad_current) < -1e-8
            if not direction_ok:
                # Если направление не обеспечивает убывания/роста, используем градиентный спуск/подъём
                p = grad_current if find_max else -grad_current
                
        except np.linalg.LinAlgError:
            # В случае ошибки решения системы (например, вырожденный гессиан)
            # Используем градиентный спуск/подъём
            p = grad_current if find_max else -grad_current
            
        # Линейный поиск для определения оптимального шага alpha
        alpha = line_search(
            f, x, p, grad_current, tracker, 
            find_max=find_max,
            c1=1e-5 if "Rosenbrock" in f.__name__ else 1e-4,  # Адаптация параметров для Rosenbrock
            c2=0.1 if "Rosenbrock" in f.__name__ else 0.9
        )
        
        # Обновление параметров
        s = alpha * p  # Вектор шага
        x_new = x + s  # Новая точка
        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad_func(x_new))
        alphas.append(alpha)
        steps.append(s)
        
        # Условия остановки
        grad_norm = np.linalg.norm(gradients[-1])  # Норма градиента
        step_norm = np.linalg.norm(s)  # Норма шага
        f_change = abs(f_values[-2] - f_values[-1])  # Изменение значения функции
        if grad_norm < eps or step_norm < 1e-10 or f_change < 1e-14:
            break
            
        x = x_new  # Обновление текущей точки
    
    # Логирование результатов
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
    logging.info(output_text)
    
    if plot:
        title = 'Newton Method (Maximization)' if find_max else 'Newton Method'
        make_plot(np.array(trajectory), f, title)
        
    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)
    
    return trajectory[-1]

# Метод DFP (Davidon-Fletcher-Powell)
def dfp_method(f: Callable, x0: np.ndarray, eps: float, tracker: OptimizationTracker, 
               grad_f: Callable = None, plot: bool = True, table: bool = True, 
               find_max: bool = False, max_iter: int = 500) -> Optional[pd.DataFrame]:
    """Метод DFP для минимизации/максимизации функции (Davidon-Fletcher-Powell).
    
    Метод использует квазиньютоновское обновление матрицы H для аппроксимации обратного гессиана.
    Поддерживает как минимизацию, так и максимизацию через флаг `find_max`.
    """
    tracker.reset()
    x = np.array(x0, dtype=float)  # Текущая точка
    n = len(x)  # Размерность задачи
    H = np.eye(n)  # Инициализация матрицы H как единичной (начальное приближение обратного гессиана)
    
    # Выбор функции для вычисления градиента
    grad_func = grad_f if grad_f else ParallelGradient(f, tracker).compute
    grad_current = grad_func(x)  # Начальный градиент
    trajectory = [x.copy()]  # Траектория итераций
    f_values = [f(x)]  # Значения целевой функции
    gradients = [grad_current.copy()]  # Градиенты на каждой итерации
    alphas = []  # Шаги линейного поиска
    steps = []  # Векторы шагов
    hessians = [H.copy()]  # Хранение матриц H
    
    for iteration in range(max_iter):
        # Сброс матрицы H каждые 5 итераций для предотвращения накопления ошибок
        if iteration % 5 == 0:
            H = np.eye(n)
        
        # Определение направления поиска
        if find_max:
            p = H @ grad_current  # Направление для максимизации
            # Если направление не обеспечивает роста, переходим к градиентному подъему
            if np.dot(p, grad_current) <= 0:
                p = grad_current
                H = np.eye(n)
        else:
            p = -H @ grad_current  # Направление для минимизации
            # Если направление не обеспечивает убывания, переходим к градиентному спуску
            if np.dot(p, grad_current) >= 0:
                p = -grad_current
                H = np.eye(n)
        
        # Линейный поиск для определения оптимального шага alpha
        alpha = line_search(
            f, x, p, grad_current, tracker, 
            find_max=find_max,
            c1=1e-5 if "Rosenbrock" in f.__name__ else 1e-4,  # Адаптация параметров для Rosenbrock
            c2=0.1 if "Rosenbrock" in f.__name__ else 0.9
        )
        
        # Обновление параметров
        s = alpha * p  # Вектор шага
        x_new = x + s  # Новая точка
        grad_new = grad_func(x_new)  # Градиент в новой точке
        y = grad_new - grad_current  # Разница градиентов
        
        # Формула обновления матрицы H (DFP-формула)
        s_vec = s.reshape(-1, 1)  # Преобразование вектора шага в столбец
        y_vec = y.reshape(-1, 1)  # Преобразование вектора y в столбец
        sy = float(s_vec.T @ y_vec)  # Скалярное произведение s и y
        
        if sy > 1e-10:  # Избегаем деления на ноль
            # DFP-обновление матрицы H:
            # H = H + (s*s^T)/s^Ty - (H*y*y^T*H)/(y^T*H*y)
            H_update = (
                (s_vec @ s_vec.T) / sy 
                - (H @ y_vec @ y_vec.T @ H) / (y_vec.T @ H @ y_vec + 1e-10)  # Добавляем эпсилон для стабильности
            )
            H = H + H_update
        
        # Проверка положительной определенности матрицы H
        if np.any(np.linalg.eigvalsh(H) < 0):  # Если есть отрицательные собственные значения
            H = np.eye(n)  # Сброс матрицы H
            
        # Обновление истории
        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad_new.copy())
        alphas.append(alpha)
        steps.append(s.flatten())
        hessians.append(H.copy())
        
        # Условия остановки
        grad_norm = np.linalg.norm(grad_new)  # Норма градиента
        step_norm = np.linalg.norm(s)  # Норма шага
        f_change = abs(f_values[-2] - f_values[-1])  # Изменение значения функции
        if grad_norm < eps or step_norm < 1e-8 or f_change < 1e-14:
            break
            
        x = x_new  # Обновление текущей точки
        grad_current = grad_new  # Обновление текущего градиента
    
    # Логирование результатов
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
    logging.info(output_text)
    
    if plot:
        make_plot(np.array(trajectory), f, 'DFP Method')
        
    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)
    
    return trajectory[-1]

# Графический интерфейс приложения
class OptimizationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Optimization Methods Comparison")
        self.tracker = OptimizationTracker()
        self.params = [3, 1, 2, 1, 1, 2, 3, 1, 2, 1]  # Параметры для многокомпонентной функции
        self.functions = [
            ("Quadratic Function", _f_quadratic, grad_quadratic, hessian_quadratic, False),
            ("Rosenbrock Function", _f_rosenbrock, grad_rosenbrock, hessian_rosenbrock, False),
            ("Multi-Exponential Function", partial(_f_multi_exponential, params=self.params), None, None, True)
        ]
        self.create_widgets()
    
    def create_widgets(self) -> None:
        """Создание элементов управления"""
        control_frame = ttk.LabelFrame(self.root, text="Parameters")
        control_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ttk.Label(control_frame, text="Function:").grid(row=0, column=0, padx=5, pady=5)
        self.func_var = tk.StringVar()
        self.func_combobox = ttk.Combobox(control_frame, textvariable=self.func_var,
                                          values=[f[0] for f in self.functions])
        self.func_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.func_combobox.current(0)
        
        ttk.Label(control_frame, text="Initial Point:").grid(row=1, column=0, padx=5, pady=5)
        self.point_var = tk.StringVar()
        self.point_combobox = ttk.Combobox(control_frame, textvariable=self.point_var,
                                          values=["[-1.2, 1]", "[0, 0]", "[2, 2]", "Custom"])
        self.point_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.point_combobox.current(0)
        
        self.custom_point_frame = ttk.Frame(control_frame)
        ttk.Label(self.custom_point_frame, text="x:").pack(side=tk.LEFT)
        self.x_entry = ttk.Entry(self.custom_point_frame, width=5)
        self.x_entry.pack(side=tk.LEFT)
        ttk.Label(self.custom_point_frame, text="y:").pack(side=tk.LEFT)
        self.y_entry = ttk.Entry(self.custom_point_frame, width=5)
        self.y_entry.pack(side=tk.LEFT)
        
        ttk.Label(control_frame, text="Tolerance:").grid(row=2, column=0, padx=5, pady=5)
        self.eps_var = tk.StringVar()
        self.eps_combobox = ttk.Combobox(control_frame, textvariable=self.eps_var,
                                      values=[f"1e-{i}" for i in range(1, 8)] + ["Custom"])
        self.eps_combobox.grid(row=2, column=1, padx=5, pady=5)
        self.eps_combobox.current(0)
        
        self.run_frame = ttk.Frame(control_frame)
        self.run_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(self.run_frame, text="Run Newton Method",
                 command=lambda: self.run_optimization('newton')).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.run_frame, text="Run DFP Method",
                 command=lambda: self.run_optimization('dfp')).pack(side=tk.LEFT, padx=5)
        
        self.result_frame = ttk.LabelFrame(self.root, text="Results")
        self.result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(self.result_frame, height=15)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.output_scroll = ttk.Scrollbar(self.result_frame, command=self.output_text.yview)
        self.output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=self.output_scroll.set)
        
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.point_combobox.bind("<<ComboboxSelected>>", self.toggle_custom_point)
        self.toggle_custom_point()
    
    def toggle_custom_point(self, event: Optional[tk.Event] = None) -> None:
        """Показ/скрытие поля ввода пользовательской начальной точки"""
        if self.point_combobox.get() == "Custom":
            self.custom_point_frame.grid(row=1, column=2, padx=5, pady=5)
        else:
            self.custom_point_frame.grid_remove()
    
    def get_parameters(self) -> Tuple[str, Callable, bool, List[float], float]:
        """Получение параметров из GUI"""
        func_idx = self.func_combobox.current()
        f_name, f, grad_f, hess_f, find_max = self.functions[func_idx]
        
        if grad_f is None:
            grad_f = ParallelGradient(f, self.tracker).compute
        if hess_f is None:
            hess_f = partial(hessian, f=f, tracker=self.tracker)
            
        f = make_function_with_counter(f, self.tracker)
        
        if self.point_combobox.get() == "Custom":
            try:
                x = float(self.x_entry.get())
                y = float(self.y_entry.get())
                if not (-1e6 <= x <= 1e6 and -1e6 <= y <= 1e6):
                    raise ValueError("Coordinates out of range")
                x0 = [x, y]
            except ValueError:
                raise ValueError("Invalid initial point coordinates")
        else:
            x0 = eval(self.point_combobox.get())  # Использование eval для строк вроде "[-1.2, 1]"
        
        if self.eps_combobox.get() == "Custom":
            try:
                eps = float(self.eps_var.get())
                if not (1e-10 <= eps <= 1e-1):
                    raise ValueError("Invalid tolerance value")
            except ValueError:
                raise ValueError("Invalid tolerance input")
        else:
            eps = float(self.eps_combobox.get().replace("1e-", "1e-"))
        
        return f_name, f, grad_f, hess_f, find_max, x0, eps
    
    def run_optimization(self, method: str) -> None:
        """Запуск оптимизации в отдельном потоке"""
        self.output_text.delete(1.0, tk.END)
        self.figure.clear()
        
        try:
            f_name, f, grad_f, hess_f, find_max, x0, eps = self.get_parameters()
        except Exception as e:
            self.output_text.insert(tk.END, f"Input Error: {str(e)}")
            return
            
        self.output_text.insert(tk.END, f"Running {method} method...\n")
        self.root.update()
        
        def optimization_thread():
            try:
                if method == 'newton':
                    result = newton_method(f, x0, eps, self.tracker, grad_f, hess_f, 
                                         plot=False, table=True, find_max=find_max)
                else:
                    result = dfp_method(f, x0, eps, self.tracker, grad_f,
                                      plot=False, table=True, find_max=find_max)
                
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
                ax.set_title(f"{method} Method\n"
                             f"Final Point: {x_store[-1].round(4)}\n"
                             f"Iterations: {len(x_store)}")
                self.canvas.draw()
                
                self.output_text.insert(tk.END, f"\nResults for {f_name}:\n")
                self.output_text.insert(tk.END, f"Iterations: {len(result)}\n")
                self.output_text.insert(tk.END, f"Function calls: {self.tracker.f_count}\n")
                self.output_text.insert(tk.END, f"Gradient calls: {self.tracker.grad_count}\n")
                self.output_text.insert(tk.END, f"Hessian calls: {self.tracker.hessian_count}\n")
                self.output_text.insert(tk.END, f"Final point: {x_store[-1].round(6)}\n")
                self.output_text.insert(tk.END, f"Final value: {result.iloc[-1]['f(x,y)'].round(6)}\n")
            except Exception as e:
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                self.output_text.insert(tk.END, error_msg)
                logging.error(error_msg)
            finally:
                self.tracker.reset()
        
        Thread(target=optimization_thread).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()