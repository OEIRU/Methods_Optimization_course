import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from functools import partial
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import warnings
from numba import cuda
import numba.cuda as nb_cuda
import time
import numba.cuda.cudadrv.driver as cuda_driver


def check_cuda_diagnostics():
    try:
        print("CUDA доступна:", cuda.is_available())
        print("Устройства:", cuda_driver.get_devices())
        print("Numba CUDA версия:", cuda.runtime.get_version())
    except Exception as e:
        print("Ошибка диагностики CUDA:", str(e))

check_cuda_diagnostics()

# Проверка наличия CUDA
CUDA_AVAILABLE = cuda.is_available()
if CUDA_AVAILABLE:
    try:
        #@cuda.jit(device=True)
        #def _test_kernel():
        #    pass
        @cuda.jit
        def _test_kernel(x):
            x[0] += 1  # Более простое ядро для тестирования


        def _test_cuda():
            x = np.array([1.0], dtype=np.float32)
            d_x = cuda.to_device(x)
            _test_kernel[1, 1](d_x)
            return d_x.copy_to_host()[0] == 2.0


    except Exception as e:
        CUDA_AVAILABLE = False
        try:
            if _test_cuda():
                CUDA_AVAILABLE = True
        except Exception as e:
            warnings.warn(f"CUDA недоступна: {str(e)}")


class OptimizationTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.f_count = 0
        self.grad_count = 0
        self.hessian_count = 0
        self.cuda_errors = 0
        self.start_time = time.time()
        self.last_update = self.start_time


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
        result = (self.f(x_plus) - self.f(x_minus)) / (2 * self.h)
        return result

    def compute(self, x):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda i: self._calc_partial(x, i),
                range(len(x))
            ))
        self.tracker.grad_count += 2 * len(x)
        return np.array(results)


# CUDA-ускоренные версии
if CUDA_AVAILABLE:
    @cuda.jit
    def _cuda_grad_kernel(f_device, x, h, grad_out):
        i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        if i < x.shape[0]:
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad_out[i] = (f_device(x_plus) - f_device(x_minus)) / (2 * h)


    @cuda.jit
    def _cuda_hessian_kernel(f_device, x, h, hess_out):
        i, j = cuda.grid(2)
        if i < x.shape[0] and j < x.shape[0]:
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            x_pp[i] += h;
            x_pp[j] += h
            x_pm[i] += h;
            x_pm[j] -= h
            x_mp[i] -= h;
            x_mp[j] += h
            x_mm[i] -= h;
            x_mm[j] -= h
            hess_out[i, j] = (f_device(x_pp) - f_device(x_pm) - f_device(x_mp) + f_device(x_mm)) / (4 * h ** 2)


    def cuda_grad(f, x, tracker, h=1e-6):
        if not CUDA_AVAILABLE:
            return grad(f, x, tracker, h)
        try:
            # Подготовка данных для GPU
            x_device = cuda.to_device(x)
            grad_out = cuda.device_array_like(x)
            n = x.shape[0]

            # Запуск ядра
            threads_per_block = min(32, n)
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            _cuda_grad_kernel[blocks_per_grid, threads_per_block](f, x_device, h, grad_out)

            # Синхронизация и получение результата
            cuda.synchronize()
            tracker.grad_count += 2 * n
            return grad_out.copy_to_host()
        except Exception as e:
            tracker.cuda_errors += 1
            warnings.warn(f"Ошибка CUDA-градиента: {str(e)}")
            return grad(f, x, tracker, h)


    def cuda_hessian(f, x, tracker, h=1e-6):
        if not CUDA_AVAILABLE:
            return hessian(f, x, tracker, h)
        try:
            # Подготовка данных для GPU
            x_device = cuda.to_device(x)
            n = x.shape[0]
            hess_out = cuda.device_array((n, n))

            # Запуск ядра
            threads_per_block = (min(8, n), min(8, n))
            blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                               (n + threads_per_block[1] - 1) // threads_per_block[1])
            _cuda_hessian_kernel[blocks_per_grid, threads_per_block](f, x_device, h, hess_out)

            # Синхронизация и получение результата
            cuda.synchronize()
            tracker.hessian_count += 4 * n * n
            return hess_out.copy_to_host()
        except Exception as e:
            tracker.cuda_errors += 1
            warnings.warn(f"Ошибка CUDA-гессиана: {str(e)}")
            return hessian(f, x, tracker, h)
else:
    def cuda_grad(f, x, tracker, h=1e-6):
        return grad(f, x, tracker, h)


    def cuda_hessian(f, x, tracker, h=1e-6):
        return hessian(f, x, tracker, h)


def make_function_with_counter(f, tracker):
    def wrapped(x):
        tracker.f_count += 1
        return f(x)

    return wrapped


# Тестовые функции
def _f_quadratic(x):
    return 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2


def _f_rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def _f_multi_exponential(x, params):
    A1, a1, b1, c1, d1, A2, a2, b2, c2, d2 = params
    term1 = A1 * np.exp(-((x[0] - a1) / b1) ** 2 - ((x[1] - c1) / d1) ** 2)
    term2 = A2 * np.exp(-((x[0] - a2) / b2) ** 2 - ((x[1] - c2) / d2) ** 2)
    return term1 + term2


# Градиент и гессиан
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
                x_pp[i] += h;
                x_pp[j] += h
                x_pm[i] += h;
                x_pm[j] -= h
                x_mp[i] -= h;
                x_mp[j] += h
                x_mm[i] -= h;
                x_mm[j] -= h
                futures.append(executor.submit(
                    lambda: (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
                ))
        for idx, future in enumerate(futures):
            i = idx // n
            j = idx % n
            hess[i, j] = future.result()
            tracker.hessian_count += 4
    return hess


# Визуализация
def make_plot(x_store, f, algorithm, params=None):
    x_store = np.array(x_store)
    padding = 0.5
    x1_min, x1_max = x_store[:, 0].min() - padding, x_store[:, 0].max() + padding
    x2_min, x2_max = x_store[:, 1].min() - padding, x_store[:, 1].max() + padding
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


# Таблица результатов
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


# Линейный поиск с коррекцией для максимизации
def line_search(f, x, p, nabla, tracker, find_max=False, c1=1e-4, c2=0.9, max_iter=100):
    alpha = 1.0
    fx = f(x)
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = f(x_new)
        grad_new = grad(f, x_new, tracker)
        if find_max:
            armijo = fx_new >= fx + c1 * alpha * np.dot(nabla, p)
            curvature = np.dot(grad_new, p) >= c2 * np.dot(nabla, p)
        else:
            armijo = fx_new <= fx + c1 * alpha * np.dot(nabla, p)
            curvature = np.dot(grad_new, p) >= c2 * np.dot(nabla, p)
        if armijo and curvature:
            return alpha
        alpha *= 0.5
    return alpha


# Метод Ньютона с регуляризацией
def newton_method(f, x0, eps, tracker, plot=True, table=True, find_max=False, max_iter=5000, use_cuda=False):
    grad_func = cuda_grad if use_cuda and CUDA_AVAILABLE else grad
    hessian_func = cuda_hessian if use_cuda and CUDA_AVAILABLE else hessian
    tracker.reset()
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    f_values = [f(x)]
    gradients = [grad_func(f, x, tracker)]
    alphas = []
    steps = []
    hessians = []
    start_time = time.time()

    for iteration in range(max_iter):
        grad_current = gradients[-1]
        grad_norm = np.linalg.norm(grad_current)

        if grad_norm < eps:
            break

        try:
            hess = hessian_func(f, x, tracker)
            hessians.append(hess.copy())

            # Регуляризация
            eigenvalues = np.linalg.eigvalsh(hess)
            min_eigen = np.min(eigenvalues)
            if min_eigen < 1e-8:
                hess += (1e-8 - min_eigen) * np.eye(len(x))

            p = np.linalg.solve(hess, -grad_current)
            if find_max:
                p = -p

            # Проверка направления
            direction_ok = np.dot(p, grad_current) > 1e-8 if find_max else np.dot(p, grad_current) < -1e-8
            if not direction_ok:
                p = grad_current if find_max else -grad_current

        except np.linalg.LinAlgError:
            # Резервный метод - градиентный спуск
            p = grad_current if find_max else -grad_current

        alpha = line_search(f, x, p, grad_current, tracker, find_max)
        s = alpha * p
        x_new = x + s

        # Сохранение результатов
        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad_func(f, x_new, tracker))
        alphas.append(alpha)
        steps.append(s)

        # Обновление параметров
        x = x_new
        tracker.last_update = time.time()

        # Проверка на сходимость
        step_norm = np.linalg.norm(s)
        if step_norm < 1e-8:
            break

    duration = time.time() - start_time

    # Формирование вывода
    output_lines = [
        "Метод Ньютона",
        f"Количество итераций: {len(trajectory)}",
        f"Время выполнения: {duration:.3f} с",
        f"Функционал вызвано: {tracker.f_count}",
        f"Градиент вызвано: {tracker.grad_count}",
        f"Гессиан вызвано: {tracker.hessian_count}",
        f"Ошибки CUDA: {tracker.cuda_errors}",
        f"Точность: {eps:.3f}",
        " i (x, y) f(x, y) S lambda угол Δ(X) Δ(Y) Δ(f) Градиент"
    ]

    for i in range(len(trajectory)):
        x_prev = trajectory[i - 1] if i > 0 else trajectory[i]
        x_curr = trajectory[i]
        f_prev = f_values[i - 1] if i > 0 else f_values[i]
        f_curr = f_values[i]
        grad_curr = gradients[i]
        alpha = alphas[i - 1] if i > 0 else 0
        s = steps[i - 1] if i > 0 else np.zeros_like(x_curr)

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
        title = 'Метод Ньютона (Максимизация)' if find_max else 'Метод Ньютона'
        make_plot(np.array(trajectory), f, title)

    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)

    return trajectory[-1]


# Метод DFP с коррекцией для максимизации
def dfp_method(f, x0, eps, tracker, plot=True, table=True, find_max=False, max_iter=500, use_cuda=False):
    grad_func = cuda_grad if use_cuda and CUDA_AVAILABLE else grad
    hessian_func = cuda_hessian if use_cuda and CUDA_AVAILABLE else hessian
    tracker.reset()
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    grad_current = grad_func(f, x, tracker)
    trajectory = [x.copy()]
    f_values = [f(x)]
    gradients = [grad_current.copy()]
    alphas = []
    steps = []
    hessians = [H.copy()]
    start_time = time.time()

    for iteration in range(max_iter):
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

        alpha = line_search(f, x, p, grad_current, tracker, find_max=find_max, c1=1e-4, c2=0.1)
        s = alpha * p
        x_new = x + s

        # Обновление матрицы H
        grad_new = grad_func(f, x_new, tracker)
        y = grad_new - grad_current
        s_vec = s.reshape(-1, 1)
        y_vec = y.reshape(-1, 1)

        s_norm = np.linalg.norm(s)
        y_norm = np.linalg.norm(y)

        if s_norm > 1e-8 and y_norm > 1e-8:
            rho = 1.0 / (y_vec.T @ s_vec + 1e-10)
            if abs(rho) < 1e10:
                Hy = H @ y_vec
                H = H - (Hy @ Hy.T) / (y_vec.T @ Hy + 1e-10) + rho * (s_vec @ s_vec.T)

        # Сохранение результатов
        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad_new.copy())
        alphas.append(alpha)
        steps.append(s.flatten())
        hessians.append(H.copy())

        # Проверка на сходимость
        grad_norm = np.linalg.norm(grad_new)
        step_norm = np.linalg.norm(s)
        f_change = abs(f_values[-2] - f_values[-1])

        if grad_norm < eps or step_norm < 1e-8 or f_change < 1e-12:
            break

        x = x_new
        grad_current = grad_new
        tracker.last_update = time.time()

    duration = time.time() - start_time

    # Формирование вывода
    output_lines = [
        "Метод DFP",
        f"Количество итераций: {len(trajectory)}",
        f"Время выполнения: {duration:.3f} с",
        f"Функционал вызвано: {tracker.f_count}",
        f"Градиент вызвано: {tracker.grad_count}",
        f"Гессиан вызвано: {tracker.hessian_count}",
        f"Ошибки CUDA: {tracker.cuda_errors}",
        f"Точность: {eps:.3f}",
        " i (x, y) f(x, y) S lambda угол Δ(X) Δ(Y) Δ(f) Градиент"
    ]

    for i in range(len(trajectory)):
        x_prev = trajectory[i - 1] if i > 0 else trajectory[i]
        x_curr = trajectory[i]
        f_prev = f_values[i - 1] if i > 0 else f_values[i]
        f_curr = f_values[i]
        grad_curr = gradients[i]
        alpha = alphas[i - 1] if i > 0 else 0
        s = steps[i - 1] if i > 0 else np.zeros_like(x_curr)

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
        make_plot(np.array(trajectory), f, 'Метод DFP')

    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)

    return trajectory[-1]


# Улучшенное GUI приложение
class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сравнение методов оптимизации")
        self.root.geometry("1200x800")

        # Стили
        self.style = ttk.Style()
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        self.style.configure("Result.TLabel", font=("Courier", 10))

        # Трекер
        self.tracker = OptimizationTracker()

        # Параметры тестовых функций
        self.params = [3, 1, 2, 1, 1, 2, 3, 1, 2, 1]

        # Функции
        self.functions = [
            ("Квадратичная функция", partial(_f_quadratic), False),
            ("Функция Розенброка", partial(_f_rosenbrock), False),
            ("Многоэкспоненциальная функция",
             partial(_f_multi_exponential, params=self.params), True)
        ]

        # История
        self.history = []

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Основные фреймы
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Левая панель управления
        control_frame = ttk.Frame(main_frame)
        main_frame.add(control_frame, weight=1)

        # Правая панель результатов
        result_frame = ttk.Frame(main_frame)
        main_frame.add(result_frame, weight=2)

        # Группа параметров
        param_group = ttk.LabelFrame(control_frame, text="Параметры оптимизации")
        param_group.pack(padx=10, pady=5, fill=tk.X)

        # Выбор функции
        ttk.Label(param_group, text="Целевая функция:").grid(row=0, column=0, padx=5, pady=5)
        self.func_var = tk.StringVar()
        self.func_combobox = ttk.Combobox(param_group, textvariable=self.func_var,
                                          values=[f[0] for f in self.functions])
        self.func_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.func_combobox.current(0)

        # Начальная точка
        ttk.Label(param_group, text="Начальная точка:").grid(row=1, column=0, padx=5, pady=5)
        self.point_var = tk.StringVar()
        self.point_combobox = ttk.Combobox(param_group, textvariable=self.point_var,
                                           values=["[-1.2, 1]", "[0, 0]", "[2, 2]", "Пользовательская"])
        self.point_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.point_combobox.current(0)

        # Пользовательская точка
        self.custom_point_frame = ttk.Frame(param_group)
        ttk.Label(self.custom_point_frame, text="x:").pack(side=tk.LEFT)
        self.x_entry = ttk.Entry(self.custom_point_frame, width=5)
        self.x_entry.pack(side=tk.LEFT)
        ttk.Label(self.custom_point_frame, text="y:").pack(side=tk.LEFT)
        self.y_entry = ttk.Entry(self.custom_point_frame, width=5)
        self.y_entry.pack(side=tk.LEFT)
        self.custom_point_frame.grid(row=1, column=2, padx=5, pady=5)

        # Точность
        ttk.Label(param_group, text="Требуемая точность:").grid(row=2, column=0, padx=5, pady=5)
        self.eps_var = tk.StringVar()
        self.eps_combobox = ttk.Combobox(param_group, textvariable=self.eps_var,
                                         values=[f"1e-{i}" for i in range(1, 8)] + ["Пользовательская"])
        self.eps_combobox.grid(row=2, column=1, padx=5, pady=5)
        self.eps_combobox.current(0)

        # Режим вычислений
        mode_group = ttk.LabelFrame(control_frame, text="Режим вычислений")
        mode_group.pack(padx=10, pady=5, fill=tk.X)

        self.compute_mode = tk.StringVar(value="cpu")

        ttk.Radiobutton(mode_group, text="Процессор (CPU)", variable=self.compute_mode,
                        value="cpu").pack(side=tk.LEFT, padx=10)

        self.cuda_radio = ttk.Radiobutton(mode_group, text="Видеокарта (GPU)",
                                          variable=self.compute_mode, value="cuda")
        self.cuda_radio.pack(side=tk.LEFT, padx=10)

        if not CUDA_AVAILABLE:
            self.cuda_radio.state(['disabled'])
            ttk.Label(mode_group, text="CUDA недоступна", foreground="red").pack(side=tk.LEFT)

        # Кнопки запуска
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Button(button_frame, text="Запустить метод Ньютона",
                   command=lambda: self.run_optimization('newton')).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Запустить метод DFP",
                   command=lambda: self.run_optimization('dfp')).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Очистить историю",
                   command=self.clear_history).pack(side=tk.RIGHT, padx=5)

        # Группа результатов
        result_group = ttk.LabelFrame(result_frame, text="Результаты")
        result_group.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Текстовый вывод
        self.output_text = tk.Text(result_group, height=15, font=("Courier", 10))
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_scroll = ttk.Scrollbar(result_group, command=self.output_text.yview)
        self.output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=self.output_scroll.set)

        # График
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # История
        history_group = ttk.LabelFrame(result_frame, text="История запусков")
        history_group.pack(padx=10, pady=5, fill=tk.X)

        self.history_listbox = tk.Listbox(history_group, height=4)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(history_group, text="Показать",
                   command=self.show_history).pack(side=tk.LEFT, padx=5)

        ttk.Button(history_group, text="Удалить",
                   command=self.delete_history).pack(side=tk.LEFT, padx=5)

        # Привязка событий
        self.point_combobox.bind("<<ComboboxSelected>>", self.toggle_custom_point)
        self.toggle_custom_point()

    def toggle_custom_point(self, event=None):
        if self.point_combobox.get() == "Пользовательская":
            self.custom_point_frame.grid()
        else:
            self.custom_point_frame.grid_remove()

    def get_parameters(self):
        func_idx = self.func_combobox.current()
        f_name, f, find_max = self.functions[func_idx]
        f = make_function_with_counter(f, self.tracker)

        # Начальная точка
        if self.point_combobox.get() == "Пользовательская":
            try:
                x = float(self.x_entry.get())
                y = float(self.y_entry.get())
                if not (-1e6 <= x <= 1e6 and -1e6 <= y <= 1e6):
                    raise ValueError("Координаты вне допустимого диапазона")
                x0 = [x, y]
            except ValueError:
                raise ValueError("Некорректный ввод координат начальной точки")
        else:
            x0 = eval(self.point_combobox.get())

        # Точность
        if self.eps_combobox.get() == "Пользовательская":
            try:
                eps = float(self.eps_var.get())
                if not (1e-10 <= eps <= 1e-1):
                    raise ValueError("Точность вне допустимого диапазона")
            except ValueError:
                raise ValueError("Некорректный ввод точности")
        else:
            eps = float(self.eps_combobox.get().replace("1e-", "1e-"))

        # Режим вычислений
        compute_mode = self.compute_mode.get()
        use_cuda = compute_mode == "cuda" and CUDA_AVAILABLE

        return f_name, f, find_max, x0, eps, use_cuda

    def clear_history(self):
        self.history = []
        self.history_listbox.delete(0, tk.END)

    def show_history(self):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            record = self.history[index]
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, record["output"])

            # Восстановление графика
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            x_store = record["x_store"]
            f = record["f"]

            X1, X2 = np.meshgrid(
                np.linspace(x_store[:, 0].min() - 0.5, x_store[:, 0].max() + 0.5, 100),
                np.linspace(x_store[:, 1].min() - 0.5, x_store[:, 1].max() + 0.5, 100))

            if "params" in record:
                Z = f([X1, X2], record["params"])
            else:
                Z = f([X1, X2])

            ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
            ax.plot(x_store[:, 0], x_store[:, 1], 'w-', lw=1)
            ax.scatter(x_store[:, 0], x_store[:, 1], c='red', edgecolors='white')
            ax.set_title(record["title"])
            self.canvas.draw()

    def delete_history(self):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            self.history.pop(index)
            self.history_listbox.delete(index)

    def run_optimization(self, method):
        self.output_text.delete(1.0, tk.END)
        self.figure.clear()
        self.canvas.draw()

        try:
            f_name, f, find_max, x0, eps, use_cuda = self.get_parameters()
        except Exception as e:
            self.output_text.insert(tk.END, f"Ошибка ввода: {str(e)}")
            return

        method_name = "Метод Ньютона" if method == 'newton' else "Метод DFP"
        self.output_text.insert(tk.END, f"Запуск {method_name}...\n{'(GPU)' if use_cuda else ''}\n")
        self.root.update()

        def optimization_thread():
            try:
                if method == 'newton':
                    result = newton_method(f, x0, eps, self.tracker, plot=False, table=True,
                                           find_max=find_max, use_cuda=use_cuda)
                else:
                    result = dfp_method(f, x0, eps, self.tracker, plot=False, table=True,
                                        find_max=find_max, use_cuda=use_cuda)

                # Сохранение в историю
                x_store = np.array([[row[1]['x'], row[1]['y']] for row in result.iterrows()])
                final_point = x_store[-1].round(6)
                final_value = result.iloc[-1]['f(x,y)'].round(6)

                title = f"{method_name} {'(GPU)' if use_cuda else ''}\nТочность: {eps:.1e}\nФункция: {f_name}"

                history_record = {
                    "output": self.output_text.get(1.0, tk.END),
                    "x_store": x_store.copy(),
                    "f": f,
                    "params": self.params.copy() if "Multi-Exponential" in f_name else None,
                    "title": title
                }

                self.history.append(history_record)
                self.history_listbox.insert(tk.END, f"{method_name} - {f_name} - {final_point}")

                # Обновление графика в основном потоке
                self.root.after(0, self.update_plot, x_store, f, f_name, title)

                # Обновление текстового вывода
                self.root.after(0, self.update_results_text, result, final_point, final_value)

            except Exception as e:
                error_msg = f"Ошибка: {str(e)}"
                self.root.after(0, self.output_text.insert, tk.END, error_msg)
            finally:
                self.root.after(0, self.update_status, "Готово")
                self.tracker.reset()

        Thread(target=optimization_thread, daemon=True).start()
        self.update_status(f"Выполняется {method_name}...")

    def update_plot(self, x_store, f, f_name, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if "Multi-Exponential" in f_name:
            plot_function = partial(_f_multi_exponential, params=self.params)
        else:
            plot_function = f

        X1, X2 = np.meshgrid(
            np.linspace(x_store[:, 0].min() - 0.5, x_store[:, 0].max() + 0.5, 100),
            np.linspace(x_store[:, 1].min() - 0.5, x_store[:, 1].max() + 0.5, 100))

        Z = plot_function([X1, X2])
        ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
        ax.plot(x_store[:, 0], x_store[:, 1], 'w-', lw=1)
        ax.scatter(x_store[:, 0], x_store[:, 1], c='red', edgecolors='white')
        ax.set_title(title)
        self.canvas.draw()

    def update_results_text(self, result, final_point, final_value):
        self.output_text.insert(tk.END, f"\nРезультаты для {result.index[-1]} итераций:\n")
        self.output_text.insert(tk.END, f"Функционал вызвано: {self.tracker.f_count}\n")
        self.output_text.insert(tk.END, f"Градиент вызвано: {self.tracker.grad_count}\n")
        self.output_text.insert(tk.END, f"Гессиан вызвано: {self.tracker.hessian_count}\n")
        if CUDA_AVAILABLE:
            self.output_text.insert(tk.END, f"Ошибки CUDA: {self.tracker.cuda_errors}\n")
        self.output_text.insert(tk.END, f"Конечная точка: {final_point}\n")
        self.output_text.insert(tk.END, f"Конечное значение: {final_value}\n")

    def update_status(self, message):
        status_bar = self.root.children.get("statusbar")
        if status_bar:
            status_bar.config(text=message)
        else:
            status_bar = ttk.Label(self.root, text=message, relief=tk.SUNKEN, name="statusbar")
            status_bar.pack(side=tk.BOTTOM, fill=tk.X)


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()