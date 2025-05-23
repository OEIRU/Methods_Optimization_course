import math
import csv
import os
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Point:
    """Класс для представления точки в двумерном пространстве."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __ne__(self, other: 'Point') -> bool:
        return self.x != other.x or self.y != other.y

    def __repr__(self) -> str:
        return f"({self.x:.6f}, {self.y:.6f})"


# Целевая функция
def f(p: Point) -> float:
    return 4 * (p.y - p.x) ** 2 + 3 * (p.x - 1) ** 2


# Штрафные функции
def penalty_area(p: Point, power: int = 1) -> float:
    return max(0, 1.0 + p.y + p.x) ** power


def barrier_area(p: Point) -> float:
    val = -(p.y + p.x + 1.0)
    if val <= 0:
        return float('inf')
    return -math.log(val)


# Объединенная целевая функция
def Q(p: Point, r: float, f: Callable, b: Callable) -> float:
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
    def safe_r(value: float) -> float:
        return min(max(value, 1e-6), 1e6)

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
    return Point(p.x + alpha * d.x, p.y + alpha * d.y)


def save_results(filename: str, results: List[List]) -> None:
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
    plt.figure()
    plt.plot(x_labels, y_data, marker='o')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class OptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод Хука-Дживса с штрафными функциями")
        self.root.geometry("1200x800")

        # Поля ввода
        self.create_input_fields()
        # Кнопки управления
        self.create_buttons()
        # Таблица результатов
        self.create_result_table()
        # График
        self.create_plot_area()

    def create_input_fields(self):
        self.params_frame = ttk.LabelFrame(self.root, text="Параметры эксперимента")
        self.params_frame.pack(padx=10, pady=10, fill="x")

        # Начальная точка
        ttk.Label(self.params_frame, text="Начальная точка (x, y):").grid(row=0, column=0, sticky="w")
        self.x0_entry = ttk.Entry(self.params_frame, width=10)
        self.y0_entry = ttk.Entry(self.params_frame, width=10)
        self.x0_entry.grid(row=0, column=1)
        self.y0_entry.grid(row=0, column=2)

        # Начальный r
        ttk.Label(self.params_frame, text="Начальный r:").grid(row=1, column=0, sticky="w")
        self.r0_entry = ttk.Entry(self.params_frame, width=10)
        self.r0_entry.grid(row=1, column=1)

        # Степени штрафной функции
        ttk.Label(self.params_frame, text="Степени штрафной функции (через запятую):").grid(row=2, column=0, sticky="w")
        self.penalty_powers_entry = ttk.Entry(self.params_frame, width=30)
        self.penalty_powers_entry.grid(row=2, column=1, columnspan=2)

        # Тип штрафа
        ttk.Label(self.params_frame, text="Тип штрафа:").grid(row=3, column=0, sticky="w")
        self.penalty_type_var = tk.StringVar(value="penalty")
        ttk.Radiobutton(self.params_frame, text="Штраф", variable=self.penalty_type_var, value="penalty").grid(row=3,
                                                                                                               column=1)
        ttk.Radiobutton(self.params_frame, text="Барьер", variable=self.penalty_type_var, value="barrier").grid(row=3,
                                                                                                                column=2)

        # Стратегии обновления r
        ttk.Label(self.params_frame, text="Стратегия обновления r:").grid(row=4, column=0, sticky="w")
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(self.params_frame, textvariable=self.strategy_var, width=30)
        self.strategy_combo["values"] = [
            "rk+1 = rk + 1",
            "rk+1 = 2*rk",
            "rk+1 = 10*rk",
            "rk+1 = 100*rk",
            "rk+1 = (rk + 1)^2",
            "rk+1 = (rk + 1)^3"
        ]
        self.strategy_combo.current(1)
        self.strategy_combo.grid(row=4, column=1, columnspan=2)

        # Точность (eps)
        ttk.Label(self.params_frame, text="Точность (eps):").grid(row=5, column=0, sticky="w")
        self.eps_entry = ttk.Entry(self.params_frame, width=10)
        self.eps_entry.grid(row=5, column=1)

    def create_buttons(self):
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.run_button = ttk.Button(self.button_frame, text="Запустить эксперимент", command=self.run_experiment)
        self.run_button.pack(side="left", padx=5)

        self.save_button = ttk.Button(self.button_frame, text="Сохранить результаты", command=self.save_results)
        self.save_button.pack(side="left", padx=5)

        self.plot_button = ttk.Button(self.button_frame, text="Построить график", command=self.plot_results)
        self.plot_button.pack(side="left", padx=5)

    def create_result_table(self):
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.pack(padx=10, fill="both", expand=True)

        self.tree = ttk.Treeview(self.table_frame, show="headings")
        self.tree["columns"] = ("x0", "y0", "r0", "iterations", "evaluations", "x_min", "y_min", "f_min")
        self.tree.column("x0", width=80)
        self.tree.column("y0", width=80)
        self.tree.column("r0", width=80)
        self.tree.column("iterations", width=80)
        self.tree.column("evaluations", width=100)
        self.tree.column("x_min", width=80)
        self.tree.column("y_min", width=80)
        self.tree.column("f_min", width=100)

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col.upper())

        self.tree.pack(fill="both", expand=True)

    def create_plot_area(self):
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_experiment(self):
        try:
            x0 = float(self.x0_entry.get())
            y0 = float(self.y0_entry.get())
            r0 = float(self.r0_entry.get())
            eps = float(self.eps_entry.get())
            penalty_powers = [int(x.strip()) for x in self.penalty_powers_entry.get().split(",")]
            strategy_name = self.strategy_var.get()
            is_barrier = self.penalty_type_var.get() == "barrier"

            strategy = recalc_strategy_factory(strategy_name)
            b_func = barrier_area if is_barrier else lambda p: penalty_area(p, power=penalty_powers[0])

            result = hooke_jeeves_penalty(f, b_func, Point(x0, y0), r0, 0.1, eps, is_barrier, strategy)

            self.tree.insert("", "end", values=(
                x0, y0, r0,
                result["iterations"], result["evaluations"],
                f"{result['x']:.6f}", f"{result['y']:.6f}", f"{result['f_min']:.6f}"
            ))

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_results(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            results = []
            for item in self.tree.get_children():
                results.append(self.tree.item(item)["values"])
            save_results(filename, results)
            messagebox.showinfo("Сохранение", f"Результаты сохранены в {filename}")

    def plot_results(self):
        self.ax.clear()
        for item in self.tree.get_children():
            values = self.tree.item(item)["values"]
            f_min = float(values[-1])
            self.ax.plot([f_min], marker='o')
        self.ax.set_title("Сходимость")
        self.ax.set_ylabel("f min")
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationGUI(root)
    root.mainloop()
