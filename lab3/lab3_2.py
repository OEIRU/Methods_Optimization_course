import math
import csv
import os
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple


class Point:
    """Класс для представления точки в двумерном пространстве."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __ne__(self, other: 'Point') -> bool:
        return self.x != other.x or self.y != other.y
    
    def __repr__(self) -> str:
        return f"({self.x:.6f}, {self.y:.6f})"


def f(p: Point) -> float:
    """Целевая функция для оптимизации."""
    return 4 * (p.y - p.x) ** 2 + 3 * (p.x - 1) ** 2


def penalty_area(p: Point, power: int = 1) -> float:
    """Штрафная функция для области ограничений."""
    return max(0, 1.0 + p.y + p.x) ** power


def penalty_line(p: Point, power: int = 1) -> float:
    """Штрафная функция для линейного ограничения."""
    return abs(p.y - p.x - 1) ** power


def barrier_area(p: Point) -> float:
    """Барьерная функция для области ограничений."""
    val = -(p.y + p.x + 1.0)
    if val <= 0:
        return float('inf')
    return -math.log(val)


def Q(p: Point, r: float, f: Callable, b: Callable) -> float:
    """Объединенная целевая функция с штрафом."""
    return f(p) + r * b(p)


def hooke_jeeves_penalty(
    f: Callable,
    b: Callable,
    p_0: Point,
    r: float,
    step: float,
    eps: float,
    is_barrier: bool,
    r_strategy: Callable
) -> Dict[str, float]:
    """
    Метод Хука-Дживса с штрафными функциями.
    
    Args:
        f: Целевая функция
        b: Штрафная/барьерная функция
        p_0: Начальная точка
        r: Начальный коэффициент штрафа
        step: Начальный шаг
        eps: Точность
        is_barrier: Флаг использования барьера
        r_strategy: Стратегия обновления коэффициента штрафа
    
    Returns:
        Результат оптимизации
    """
    p = p_0
    count_i = 0
    count_f = 0
    r_values = []
    
    while step > eps:
        for direction in [Point(1, 0), Point(0, 1)]:
            p_test_plus = Point(p.x + step * direction.x, p.y + step * direction.y)
            p_test_minus = Point(p.x - step * direction.x, p.y - step * direction.y)
            count_f += 4
            
            if Q(p_test_plus, r, f, b) < Q(p, r, f, b):
                count_f -= 2
                p = p_test_plus
            elif Q(p_test_minus, r, f, b) < Q(p, r, f, b):
                p = p_test_minus
        
        if p_0 != p:
            dir = Point(p.x - p_0.x, p.y - p_0.y)
            start, end, count_f1 = interval(f, b, p_0, dir, r, eps / 2.0, 1.0)
            lam, _, count_f2 = golden_ratio(f, b, p_0, dir, r, start, end, eps)
            p = Point(p_0.x + lam * dir.x, p_0.y + lam * dir.y)
            count_f += count_f1 + count_f2
        else:
            step /= 2.0
        
        p_0 = p
        r = r_strategy(r, b, p, eps, is_barrier)
        r_values.append(r)
        count_i += 1
    
    return {
        'x': p.x,
        'y': p.y,
        'f_min': f(p),
        'iterations': count_i,
        'evaluations': count_f,
        'r_values': r_values
    }


def recalc_strategy_factory(name: str) -> Callable:
    """Фабрика стратегий обновления коэффициента штрафа."""
    def safe_r(value: float) -> float:
        return min(max(value, 1e-6), 1e6)  # Ограничиваем r от 1e-6 до 1e6
    
    strategies = {
        "rk+1 = rk + 1": lambda r, b, p, eps, is_barrier: safe_r(r + 1),
        "rk+1 = 2*rk": lambda r, b, p, eps, is_barrier: safe_r(r * 2),
        "rk+1 = 10*rk": lambda r, b, p, eps, is_barrier: safe_r(r * 10),
        "rk+1 = 100*rk": lambda r, b, p, eps, is_barrier: safe_r(r * 100),
        "rk+1 = (rk + 1)^2": lambda r, b, p, eps, is_barrier: safe_r((r + 1) ** 2),
        "rk+1 = (rk + 1)^3": lambda r, b, p, eps, is_barrier: safe_r((r + 1) ** 3)
    }
    
    return strategies.get(name, lambda r, b, p, eps, is_barrier: safe_r((r / 2) if is_barrier else r * 2))


def interval(
    f: Callable,
    b: Callable,
    p: Point,
    direction: Point,
    r: float,
    delta: float,
    x0: float
) -> Tuple[float, float, int]:
    """
    Поиск интервала для минимизации.
    
    Args:
        f: Целевая функция
        b: Штрафная функция
        p: Начальная точка
        direction: Направление поиска
        r: Коэффициент штрафа
        delta: Шаг поиска
        x0: Начальное значение
    
    Returns:
        Найденный интервал и количество вычислений функции
    """
    count_f = 2
    h = -delta
    
    if Q(shift(p, direction, x0), r, f, b) > Q(shift(p, direction, x0 + delta), r, f, b):
        h = delta
    
    x = x0 + h
    while Q(shift(p, direction, x), r, f, b) > Q(shift(p, direction, x + 2 * h), r, f, b):
        h *= 2
        x += h
        count_f += 2
    
    return x - h, x + 2 * h, count_f


def golden_ratio(
    f: Callable,
    b: Callable,
    p: Point,
    direction: Point,
    r: float,
    start: float,
    end: float,
    eps: float
) -> Tuple[float, int, int]:
    """
    Метод золотого сечения для одномерной оптимизации.
    
    Args:
        f: Целевая функция
        b: Штрафная функция
        p: Точка в пространстве
        direction: Направление поиска
        r: Коэффициент штрафа
        start: Начало интервала
        end: Конец интервала
        eps: Точность
    
    Returns:
        Найденный минимум, количество итераций и вычислений функции
    """
    phi = (3 - math.sqrt(5)) / 2
    count_i = 0
    count_f = 2
    
    x1 = start + phi * (end - start)
    x2 = end - phi * (end - start)
    f1 = Q(shift(p, direction, x1), r, f, b)
    f2 = Q(shift(p, direction, x2), r, f, b)
    
    while abs(end - start) > eps:
        if f1 > f2:
            start, x1, f1 = x1, x2, f2
            x2 = end - phi * (end - start)
            f2 = Q(shift(p, direction, x2), r, f, b)
        else:
            end, x2, f2 = x2, x1, f1
            x1 = start + phi * (end - start)
            f1 = Q(shift(p, direction, x1), r, f, b)
        
        count_i += 1
        count_f += 1
    
    return (start + end) / 2, count_i + 1, count_f


def shift(p: Point, d: Point, alpha: float) -> Point:
    """Сдвиг точки в заданном направлении."""
    return Point(p.x + alpha * d.x, p.y + alpha * d.y)


def save_results(filename: str, results: List[List]) -> None:
    """
    Сохранение результатов в CSV файл.
    
    Args:
        filename: Имя файла для сохранения
        results: Результаты оптимизации
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "x0", "y0", "r0", "Iterations", "Evaluations", "x_min", "y_min", "f_min"
        ])
        for res in results:
            writer.writerow(res)


def plot_convergence(
    title: str,
    y_data: List[float],
    x_labels: List[str],
    ylabel: str,
    filename: str
) -> None:
    """
    Построение графика сходимости.
    
    Args:
        title: Заголовок графика
        y_data: Данные для оси Y
        x_labels: Подписи для оси X
        ylabel: Подпись для оси Y
        filename: Имя файла для сохранения графика
    """
    plt.figure()
    plt.plot(x_labels, y_data, marker='o')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def penalty_function_factory(power: int) -> Callable:
    """Фабрика штрафных функций."""
    return lambda p: max(0, 1.0 + p.y + p.x) ** power


def run_all_experiments() -> None:
    """Запуск всех экспериментов по оптимизации."""
    os.makedirs("results", exist_ok=True)
    
    # Параметры экспериментов
    penalty_powers = [1, 2, 4]
    initial_rs = [0.01, 0.1, 1.0, 10.0, 100.0]
    strategies = [
        "rk+1 = rk + 1", "rk+1 = 2*rk", "rk+1 = 10*rk",
        "rk+1 = 100*rk", "rk+1 = (rk + 1)^2", "rk+1 = (rk + 1)^3"
    ]
    initial_points = [Point(8, -10), Point(0, -6), Point(-1, -1), Point(5, -10)]
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    
    for func_type, barrier in [("penalty_area", False), ("barrier_area", True)]:
        b_func = (lambda p: penalty_area(p)) if not barrier else barrier_area
        default_strategy = recalc_strategy_factory("rk+1 = 2*rk")
        
        # Влияние самой штрафной функции
        results = []
        p0 = Point(8, -10)
        r0 = 1.0 
        strategy = recalc_strategy_factory("rk+1 = 2*rk")
        
        for power in penalty_powers:
            b_func = lambda p: penalty_area(p, power=power)
            res = hooke_jeeves_penalty(f, b_func, p0, r0, 0.1, 1e-4, False, strategy)
            results.append([
                p0.x, p0.y, r0, res['iterations'], res['evaluations'], res['x'],
                res['y'], res['f_min']
            ])
        
        filename = "results/penalty_function_variation.csv"
        save_results(filename, results)
        plot_convergence(
            "Penalty function power effect", 
            [r[-1] for r in results], 
            ["power=1", "power=2", "power=4"], 
            "f min", 
            filename.replace(".csv", ".png")
        )
        
        # Влияние начального r0
        results = []
        for r0 in initial_rs:
            p0 = Point(8, -10)
            res = hooke_jeeves_penalty(f, b_func, p0, r0, 0.1, 1e-4, barrier, default_strategy)
            results.append([
                p0.x, p0.y, r0, res['iterations'], res['evaluations'], res['x'],
                res['y'], res['f_min']
            ])
        
        filename = f"results/{func_type}_initial_r.csv"
        save_results(filename, results)
        plot_convergence(
            f"{func_type}: min от r0", 
            [r[-1] for r in results], 
            [str(r[2]) for r in results], 
            "f min", 
            filename.replace(".csv", ".png")
        )
        
        # Влияние стратегий
        results = []
        for strategy in strategies:
            p0 = Point(8, -10)
            r0 = 1.0
            strat = recalc_strategy_factory(strategy)
            res = hooke_jeeves_penalty(f, b_func, p0, r0, 0.1, 1e-4, barrier, strat)
            results.append([
                p0.x, p0.y, strategy, res['iterations'], res['evaluations'],
                res['x'], res['y'], res['f_min']
            ])
        
        filename = f"results/{func_type}_strategies.csv"
        save_results(filename, results)
        plot_convergence(
            f"{func_type}: min от стратегии", 
            [r[-1] for r in results], 
            [r[2] for r in results], 
            "f min", 
            filename.replace(".csv", ".png")
        )
        
        # Влияние начального приближения
        results = []
        for p0 in initial_points:
            r0 = 1.0
            res = hooke_jeeves_penalty(f, b_func, p0, r0, 0.1, 1e-4, barrier, default_strategy)
            results.append([
                p0.x, p0.y, r0, res['iterations'], res['evaluations'], res['x'],
                res['y'], res['f_min']
            ])
        
        filename = f"results/{func_type}_initial_x.csv"
        save_results(filename, results)
        plot_convergence(
            f"{func_type}: min от x0", 
            [r[-1] for r in results], 
            [f"({r[0]}, {r[1]})" for r in results], 
            "f min", 
            filename.replace(".csv", ".png")
        )
        
        # Влияние эпсилон
        results = []
        for eps in epsilons:
            p0 = Point(8, -10)
            r0 = 1.0
            res = hooke_jeeves_penalty(f, b_func, p0, r0, 0.1, eps, barrier, default_strategy)
            results.append([
                p0.x, p0.y, eps, res['iterations'], res['evaluations'], res['x'],
                res['y'], res['f_min']
            ])
        
        filename = f"results/{func_type}_eps.csv"
        save_results(filename, results)
        plot_convergence(
            f"{func_type}: min от eps", 
            [r[-1] for r in results], 
            [str(r[2]) for r in results], 
            "f min", 
            filename.replace(".csv", ".png")
        )


if __name__ == "__main__":
    run_all_experiments()