import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial


class OptimizationTracker:
    def __init__(self):
        self.f_count = 0
        self.grad_count = 0
        self.hessian_count = 0

    def reset(self):
        self.f_count = 0
        self.grad_count = 0
        self.hessian_count = 0


# Objective functions with tracking
def make_function_with_counter(f, tracker):
    def wrapped(x):
        tracker.f_count += 1
        return f(x)

    return wrapped


# Original functions
def _f_quadratic(x):
    return 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2


def _f_rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def _f_multi_exponential(x, params):
    A1, a1, b1, c1, d1, A2, a2, b2, c2, d2 = params
    term1 = A1 * np.exp(-((x[0] - a1) / b1) ** 2 - ((x[1] - c1) / d1) ** 2)
    term2 = A2 * np.exp(-((x[0] - a2) / b2) ** 2 - ((x[1] - c2) / d2) ** 2)
    return term1 + term2


# Visualization
def make_plot(x_store, f, algorithm, params=None):
    x_store = np.array(x_store)
    if len(x_store) == 0:
        print("No data to plot")
        return

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
            row['H'] = '\n' + str(np.round(H_store[i], 3))
        else:
            row['H'] = '-'

        data.append(row)

    return pd.DataFrame(data).set_index('Iteration')


def grad(f, x, tracker, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        tracker.grad_count += 2  # Two function calls per gradient component
    return grad


def hessian(f, x, tracker, h=1e-6):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            x_pp[i] += h
            x_pp[j] += h
            x_pm[i] += h
            x_pm[j] -= h
            x_mp[i] -= h
            x_mp[j] += h
            x_mm[i] -= h
            x_mm[j] -= h
            hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
            tracker.hessian_count += 4  # Four function calls per Hessian component
    return hess


def line_search(f, x, p, nabla, tracker, find_max=False, max_iter=100):
    alpha = 1.0
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)

    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = f(x_new)
        grad_new = grad(f, x_new, tracker)

        armijo = fx_new <= fx + c1 * alpha * np.dot(nabla, p) if not find_max else fx_new >= fx + c1 * alpha * np.dot(
            nabla, p)
        curvature = np.dot(grad_new, p) >= c2 * np.dot(nabla, p) if not find_max else np.dot(grad_new,
                                                                                             p) <= c2 * np.dot(nabla, p)

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
                direction_ok = np.dot(p, grad_current) > 1e-8
            else:
                direction_ok = np.dot(p, grad_current) < -1e-8
            if not direction_ok:
                p = grad_current if find_max else -grad_current
        except np.linalg.LinAlgError:
            p = grad_current if find_max else -grad_current

        alpha = line_search(f, x, p, grad_current, tracker, find_max)
        if alpha < 1e-10:
            alpha = 1e-4

        s = alpha * p
        x_new = x + s

        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad(f, x_new, tracker))
        alphas.append(alpha)
        steps.append(s)

        if np.linalg.norm(gradients[-1]) < eps:
            break
        x = x_new

    if plot:
        make_plot(np.array(trajectory), f, 'Newton Method')

    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)
    return trajectory[-1]


def dfp_method(f, x0, eps, tracker, plot=True, table=True, max_iter=500):
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
        p = -H @ grad_current
        alpha = line_search(f, x, p, grad_current, tracker)
        s = alpha * p
        x_new = x + s
        grad_new = grad(f, x_new, tracker)
        y = grad_new - grad_current

        if np.linalg.norm(y) > 1e-10:
            s = s.reshape(-1, 1)
            y = y.reshape(-1, 1)

            rho = 1 / (y.T @ s + 1e-10)
            Hy = H @ y
            H = H - rho * (Hy @ s.T + s @ Hy.T) + rho ** 2 * (y.T @ Hy) * s @ s.T + rho * s @ s.T

        trajectory.append(x_new.copy())
        f_values.append(f(x_new))
        gradients.append(grad_new.copy())
        alphas.append(alpha)
        steps.append(s.flatten())
        hessians.append(H.copy())

        if np.linalg.norm(grad_new) < eps:
            break
        x = x_new
        grad_current = grad_new

    if plot:
        make_plot(np.array(trajectory), f, 'DFP Method')

    if table:
        return make_table(trajectory, f_values, gradients, alphas, steps, hessians)
    return trajectory[-1]


def get_user_choice(options, prompt):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input("Ваш выбор: ")) - 1
            if 0 <= choice < len(options):
                return choice
            print("Пожалуйста, введите число из списка")
        except ValueError:
            print("Пожалуйста, введите число")


def main():
    tracker = OptimizationTracker()

    # Parameters for the multi-exponential function
    params = [3, 1, 2, 1, 1, 2, 3, 1, 2, 1]

    # Function selection
    functions = [
        ("Квадратичная функция", partial(_f_quadratic), False),
        ("Функция Розенброка", partial(_f_rosenbrock), False),
        ("Многоэкспоненциальная функция", partial(_f_multi_exponential, params=params), True)
    ]
    func_choice = get_user_choice([f[0] for f in functions], "\nВыберите функцию:")
    f_name, f, find_max = functions[func_choice]
    f = make_function_with_counter(f, tracker)

    # Starting point selection
    start_points = [
        ("[-1.2, 1]", [-1.2, 1]),
        ("[0, 0]", [0, 0]),
        ("[2, 2]", [2, 2]),
        ("Ввести свою точку", None)
    ]
    point_choice = get_user_choice([p[0] for p in start_points], "\nВыберите начальную точку:")
    if point_choice == 3:
        x = float(input("Введите x: "))
        y = float(input("Введите y: "))
        x0 = [x, y]
    else:
        x0 = start_points[point_choice][1]

    # Precision selection
    precisions = [f"1e-{i}" for i in range(1, 8)] + ["Ввести свою точность"]
    eps_choice = get_user_choice(precisions, "\nВыберите точность:")
    if eps_choice == 7:
        eps = float(input("Введите точность (например, 1e-6): "))
    else:
        eps = 10 ** -(eps_choice + 1)

    # Run optimizations
    print(f"\nРезультаты для функции {f_name}, начальная точка {x0}, точность {eps}")

    print("\nМетод Ньютона:")
    newton_result = newton_method(f, x0, eps, tracker, find_max=find_max)

    print("\nМетод DFP:")
    dfp_result = dfp_method(f, x0, eps, tracker)

    # Comparison
    print("\nСравнение методов:")
    print(
        f"{'Метод':<10} {'Функция':<25} {'Итерации':<10} {'Вызовы f':<10} {'Вызовы ∇f':<10} {'Вызовы H':<10} {'Результат':<15} {'Точка':<20}")
    print("-" * 100)

    for method_name, result in [('Newton', newton_result), ('DFP', dfp_result)]:
        if isinstance(result, pd.DataFrame):
            iterations = len(result)
            final_value = result['f(x,y)'].iloc[-1]
            final_point = f"({result['x'].iloc[-1]:.6f}, {result['y'].iloc[-1]:.6f})"
        else:
            iterations = 'N/A'
            final_value = f(result)
            final_point = f"({result[0]:.6f}, {result[1]:.6f})"

        print(
            f"{method_name:<10} {f_name[:25]:<25} {iterations:<10} {tracker.f_count:<10} {tracker.grad_count:<10} {tracker.hessian_count:<10} {final_value:<15.6f} {final_point:<20}")
        tracker.reset()


if __name__ == '__main__':
    main()