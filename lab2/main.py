import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Global variable for counting function evaluations
f_count = 0

# Objective functions
def f_quadratic(x):
    return 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2

def f_rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def f_multi_exponential(x, params):
    A1, a1, b1, c1, d1, A2, a2, b2, c2, d2 = params
    term1 = A1 * np.exp(-((x[0] - a1)/b1)**2 - ((x[1] - c1)/d1)**2)
    term2 = A2 * np.exp(-((x[0] - a2)/b2)**2 - ((x[1] - c2)/d2)**2)
    return -(term1 + term2)

# Visualization
def make_plot(x_store, f, algorithm):
    x1 = np.linspace(min(x_store[:,0])-0.5, max(x_store[:,0])+0.5, 30)
    x2 = np.linspace(min(x_store[:,1])-0.5, max(x_store[:,1])+0.5, 30)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])
    plt.figure(figsize=(10,8))
    plt.title(f"{algorithm}\nEXTREMUM: {x_store[-1]}\n{len(x_store)} ITERATIONS")
    plt.contourf(X1, X2, Z, 100, cmap='pink')
    plt.colorbar()
    plt.scatter(x_store[:,0], x_store[:,1], c='grey', zorder=3, s=50, edgecolors='w')
    plt.plot(x_store[:,0], x_store[:,1], c='w')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Create a table
def make_table(x_store, f_store, nabla_store, a_store, s_store, H_store=None):
    angle = np.arccos((x_store[:,0] * np.array(nabla_store)[:,0] +
                      x_store[:,1] * np.array(nabla_store)[:,1]) /
                      (np.sqrt(x_store[:,0]**2 + x_store[:,1]**2) *
                       np.sqrt(np.array(nabla_store)[:,0]**2 + np.array(nabla_store)[:,1]**2)))
    nabla_store = [', '.join(item) for item in np.around(np.array(nabla_store),3).astype(str)]
    s_store = [', '.join(item) for item in np.around(np.array(s_store),3).astype(str)]
    x_diff = np.insert(np.absolute(x_store[1:,0] - x_store[:-1,0]), 0, 0.)
    y_diff = np.insert(np.absolute(x_store[1:,1] - x_store[:-1,1]), 0, 0.)
    f_store = np.array(f_store)
    f_diff = np.insert(np.absolute(f_store[1:] - f_store[:-1]), 0, np.absolute(f_store[0]))

    if H_store is None:
        H_store = ['-' for _ in range(len(x_store))]
    else:
        H_store = np.around(np.reshape(np.array(H_store), (1, len(H_store), 4))[0], 1)
        H_store = [str(h).replace('\n', '') for h in H_store]

    df = pd.DataFrame({
        'x': x_store[:,0],
        'y': x_store[:,1],
        'f(x, y)': f_store,
        '(s1, s2)': s_store,
        'a': a_store,
        'x_diff': x_diff,
        'y_diff': y_diff,
        'f_diff': f_diff,
        'angle': angle,
        'grad': nabla_store,
        'hessian': H_store
    })

    df.index += 1
    return df

# Newton's method
def newton_method(f, x0, eps, plot=True, table=False):
    global f_count
    f_store, nabla_store, a_store, s_store, H_store = [], [], [], [], []
    it, f_count = 1, 0
    x = np.array(x0, dtype=float)
    x_store = [x.copy()]
    f_val = f(x)
    f_store.append(f_val)
    nabla = grad(f, x)
    H = hessian(f, x)
    nabla_store.append(nabla)
    H_store.append(H)
    a_store.append(0.0)
    s_store.append(np.zeros_like(x))

    max_iter = 5000
    while np.linalg.norm(nabla) > eps and it < max_iter:
        try:
            p = np.linalg.solve(H, -nabla)
        except np.linalg.LinAlgError:
            H += 1e-6 * np.eye(len(x))
            p = np.linalg.solve(H, -nabla)

        a = line_search(f, x, p, nabla)
        x_new = x + a * p

        nabla_new = grad(f, x_new)
        H_new = hessian(f, x_new)

        s_store.append(a*p)
        a_store.append(a)
        x_store.append(x_new.copy())
        f_store.append(f(x_new))
        nabla_store.append(nabla_new)
        H_store.append(H_new)

        x = x_new
        nabla = nabla_new
        H = H_new
        it += 1

    if len(x_store) != len(s_store):
        x_store = x_store[:-1]
        f_store = f_store[:-1]
        nabla_store = nabla_store[:-1]
        H_store = H_store[:-1]

    if plot:
        make_plot(np.array(x_store), f, 'NEWTON')
    if table:
        df = make_table(np.array(x_store), f_store, nabla_store, a_store, s_store, H_store)
        print("\nТаблица метода Ньютона:\n", df.head())
        return df

# DFP method
def pearson_method(f, start_point, eps, plot=True, table=False):
    global f_count
    x = np.array(start_point, dtype=float)
    n = len(x)
    H = np.eye(n)
    nabla = grad(f, x)
    x_store = [x.copy()]
    f_store = [f(x)]
    nabla_store = [nabla.copy()]
    s_store = [np.zeros_like(x)]
    a_store = [0.0]
    H_store = [H.copy()]
    it = 0
    max_iter = 500

    while np.linalg.norm(nabla) > eps and it < max_iter:
        p = -H @ nabla
        a = line_search(f, x, p, nabla)
        s = a * p
        x_new = x + s
        nabla_new = grad(f, x_new)
        y = nabla_new - nabla

        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)
        if (y.T @ s)[0][0] > 0:
            H += (s @ s.T)/(y.T @ s)[0][0] - (H @ y @ y.T @ H)/(y.T @ H @ y)[0][0]

        x_store.append(x_new.copy())
        f_store.append(f(x_new))
        nabla_store.append(nabla_new.copy())
        s_store.append(s.flatten().copy())
        a_store.append(a)
        H_store.append(H.copy())

        x = x_new
        nabla = nabla_new
        it += 1

    min_len = min(len(x_store), len(f_store), len(nabla_store), len(s_store), len(a_store), len(H_store))
    x_store = x_store[:min_len]
    f_store = f_store[:min_len]
    nabla_store = nabla_store[:min_len]
    s_store = s_store[:min_len]
    a_store = a_store[:min_len]
    H_store = H_store[:min_len]

    if plot:
        make_plot(np.array(x_store), f, 'PEARSON (DFP)')
    if table:
        df = make_table(np.array(x_store), f_store, nabla_store, a_store, s_store, H_store)
        print("\nТаблица метода Пирсона:\n", df.head())
        return df

# Line search
def line_search(f, x, p, nabla):
    global f_count
    a = 1.0
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    f_count += 1

    while True:
        x_new = x + a * p
        fx_new = f(x_new)
        nabla_new = grad(f, x_new)

        if fx_new <= fx + c1*a*np.dot(nabla, p) and np.dot(nabla_new, p) >= c2*np.dot(nabla, p):
            break

        a *= 0.5
        f_count += 2

        if a < 1e-10:
            break

    return a

# Gradient calculation
def grad(f, x):
    global f_count
    h = 1e-6
    d = len(x)
    nabla = np.zeros(d)
    for i in range(d):
        x_for = x.copy()
        x_back = x.copy()
        x_for[i] += h
        x_back[i] -= h
        nabla[i] = (f(x_for) - f(x_back)) / (2*h)
        f_count += 2
    return nabla

# Hessian calculation
def hessian(f, x):
    global f_count
    h = 1e-6
    d = len(x)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
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
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h**2)
            f_count += 4
    return H

# Main function
def main():
    # Parameters for the multi-exponential function
    params = [3, 1, 2, 1, 1, 2, 3, 1, 2, 1]

    # Select the starting point
    print("Выберите начальную точку:")
    print("1. [-1.2, 1]")
    print("2. [0, 0]")
    print("3. [2, 2]")
    choice_x0 = int(input("Введите номер (1-3): ") or 1)
    x0 = {
        1: [-1.2, 1],
        2: [0, 0],
        3: [2, 2]
    }.get(choice_x0, [-1.2, 1])

    # Select the precision
    print("\nВыберите точность (epsilon):")
    for i in range(1, 8):
        print(f"{i}. 1e-{i}")
    choice_eps = int(input("Введите номер (1-7): ") or 6)
    eps = 10**(-choice_eps) if 1 <= choice_eps <=7 else 1e-6

    # Run tests
    print("\nТест 1: Квадратичная функция")
    df_newton_quad = newton_method(f_quadratic, x0, eps, plot=True, table=True)
    df_pearson_quad = pearson_method(f_quadratic, x0, eps, plot=True, table=True)

    print("\nТест 2: Функция Розенброка")
    df_newton_rosen = newton_method(f_rosenbrock, x0, eps, plot=True, table=True)
    df_pearson_rosen = pearson_method(f_rosenbrock, x0, eps, plot=True, table=True)

    print("\nТест 3: Многоэкспоненциальная функция")
    df_newton_exp = newton_method(lambda x: f_multi_exponential(x, params), x0, eps, plot=True, table=True)
    df_pearson_exp = pearson_method(lambda x: f_multi_exponential(x, params), x0, eps, plot=True, table=True)

    # Create a comparison table
    comparison_data = {
        'Method': [],
        'Function': [],
        'Iterations': [],
        'Function Calls': [],
        'Final f(x)': [],
        'Final Point': []
    }

    # Add data for each method and function
    for method_name, df_quad, df_rosen, df_exp in [
        ('Newton', df_newton_quad, df_newton_rosen, df_newton_exp),
        ('Pearson (DFP)', df_pearson_quad, df_pearson_rosen, df_pearson_exp)
    ]:
        for func_name, df in [
            ('Quadratic', df_quad),
            ('Rosenbrock', df_rosen),
            ('Multi-Exponential', df_exp)
        ]:
            comparison_data['Method'].append(method_name)
            comparison_data['Function'].append(func_name)
            comparison_data['Iterations'].append(len(df))
            comparison_data['Function Calls'].append(df['f(x, y)'].count())
            comparison_data['Final f(x)'].append(df['f(x, y)'].iloc[-1])
            final_point = f"({df['x'].iloc[-1]:.5f}, {df['y'].iloc[-1]:.5f})"
            comparison_data['Final Point'].append(final_point)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    print("\nСравнение методов:\n", comparison_df)

if __name__ == "__main__":
    main()
