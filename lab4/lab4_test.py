import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm

# Целевая функция из варианта 5
def target_function(x):
    C = [1, 2, 10, 5, 7, 9]
    a = [0, 0, 3, -7, 6, 6]
    b = [-1, -4, -2, -6, -10, 1]
    
    result = 0
    for i in range(6):
        denominator = 1 + (x[0] - a[i])**2 + (x[1] - b[i])**2
        result += C[i] / denominator
    return -result  # Инвертируем для минимизации

# 1. Метод простого случайного поиска
def simple_random_search(bounds, trials=1000):
    best_point = None
    best_value = float('inf')
    convergence = []
    
    for _ in range(trials):
        point = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        value = target_function(point)
        
        if value < best_value:
            best_value = value
            best_point = point.copy()
        
        convergence.append(-best_value)
    
    return best_point, -best_value, convergence

# 2. Алгоритм 1
def algorithm_1(bounds, m=5):
    best_point = None
    best_value = float('inf')
    no_improvement = 0
    convergence = []
    
    while no_improvement < m:
        # Случайная начальная точка
        point = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        # Локальная оптимизация
        res = minimize(target_function, point, bounds=bounds)
        
        if res.success:
            local_min = res.x
            local_value = res.fun
            
            if local_value < best_value:
                best_value = local_value
                best_point = local_min.copy()
                no_improvement = 0
            else:
                no_improvement += 1
                
            convergence.append(-best_value)
        else:
            no_improvement += 1  # Увеличиваем счетчик, если оптимизация не удалась
    
    return best_point, -best_value, convergence

# 3. Алгоритм 2
def algorithm_2(bounds, m=5):
    # Начинаем с первой точки
    point = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    res = minimize(target_function, point, bounds=bounds)
    best_point = res.x
    best_value = res.fun
    no_improvement = 0
    convergence = [-best_value]
    
    while no_improvement < m:
        # Ненаправленный случайный поиск
        found_better = False
        for _ in range(200):  # Ограничение на число попыток
            point = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            if target_function(point) < best_value:
                found_better = True
                break
                
        if not found_better:
            no_improvement += 1
            convergence.append(-best_value)  # Добавляем текущее значение даже без улучшения
            continue
            
        # Локальная оптимизация
        res = minimize(target_function, point, bounds=bounds)
        local_min = res.x
        local_value = res.fun
        
        if res.success and local_value < best_value:
            best_value = local_value
            best_point = local_min.copy()
            no_improvement = 0
        else:
            no_improvement += 1
        
        convergence.append(-best_value)
    
    return best_point, -best_value, convergence

# 4. Алгоритм 3
def algorithm_3(bounds, m=5):
    # Начальная точка
    point = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    res = minimize(target_function, point, bounds=bounds)
    best_point = res.x
    best_value = res.fun
    no_improvement = 0
    convergence = [-best_value]
    
    max_direction_attempts = 1000  # Максимальное количество попыток в направлении
    attempt = 0
    
    while no_improvement < m and attempt < max_direction_attempts:
        # Случайное направление
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)
        
        # Увеличенный шаг
        step = 1.0
        new_point = best_point.copy()
        improved = False
        max_steps = 20  # Ограничиваем количество шагов в направлении
        
        for _ in range(max_steps):
            new_point = new_point + step * direction
            
            # Проверка границ
            if any(new_point[i] < bounds[i][0] or new_point[i] > bounds[i][1] 
                  for i in range(2)):
                break
                
            value = target_function(new_point)
            if value < best_value:
                # Локальная оптимизация
                res = minimize(target_function, new_point, bounds=bounds)
                if res.success:
                    local_min = res.x
                    local_value = res.fun
                    
                    if local_value < best_value:
                        best_value = local_value
                        best_point = local_min.copy()
                        no_improvement = 0
                        convergence.append(-best_value)
                        improved = True
                break
            else:
                no_improvement += 1
        
        if not improved:
            no_improvement += 1
            convergence.append(-best_value)
            
        attempt += 1
        
        # Условие выхода по расстоянию
        if np.linalg.norm(new_point - best_point) > 20:
            print("Алгоритм 3 завершен из-за большого расстояния.")
            break
                
    return best_point, -best_value, convergence

# Исследование метода простого случайного поиска
def study_simple_search():
    print("Исследование метода простого случайного поиска:")
    print("------------------------------------------------")
    print(f"{'ε':<6} | {'P':<6} | {'N':<6} | {'x':<15} | {'y':<10} | {'Значение'}")
    print("------------------------------------------------")
    
    eps_values = [0.1, 0.05, 0.01]
    P_values = [0.8, 0.9, 0.95]
    bounds = [(-10, 10), (-10, 10)]
    
    for eps in eps_values:
        for P in P_values:
            # Расчет объема области и окрестности
            V = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
            # Если ε — относительный размер окрестности по каждой оси
            V_eps = (eps * (bounds[0][1] - bounds[0][0])) * (eps * (bounds[1][1] - bounds[1][0]))
            P_eps = V_eps / V
            
            # Необходимое число испытаний
            required_N = int(np.log(1 - P) / np.log(1 - P_eps)) if P_eps < 1 else 1000
            
            # Запуск алгоритма
            point, value, _ = simple_random_search(bounds, required_N)
            
            # Вывод результатов
            print(f"{eps:<6.2f} | {P:<6.2f} | {required_N:<6} | {point[0]:<7.3f}, {point[1]:<7.3f} | {value:<10.6f}")

# Визуализация
def visualize_results(results):
    # График сходимости
    plt.figure(figsize=(12, 5))
    for name, _, conv in results:
        plt.plot(conv, label=name)
    plt.title('График сходимости')
    plt.xlabel('Итерация')
    plt.ylabel('Значение функции')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Карта уровня функции
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = -target_function([X[i, j], Y[i, j]])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap=cm.viridis)
    plt.colorbar(contour, label='Значение функции')
    
    # Отображение найденных максимумов
    for name, (x, y), _ in results:
        plt.scatter(x, y, s=100, marker='x', linewidth=2, label=f'Максимум ({name})')
    
    plt.title('Карта уровня целевой функции')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Основная функция
def main():
    bounds = [(-10, 10), (-10, 10)]
    
    # Исследование метода простого случайного поиска
    study_simple_search()
    
    # Запуск всех алгоритмов
    print("\nВыполнение алгоритмов...")
    sr_point, sr_value, sr_conv = simple_random_search(bounds, 1000)
    a1_point, a1_value, a1_conv = algorithm_1(bounds)
    a2_point, a2_value, a2_conv = algorithm_2(bounds)
    a3_point, a3_value, a3_conv = algorithm_3(bounds)
    
    # Вывод результатов
    print("\nРезультаты оптимизации:")
    print("------------------------------------------------")
    print(f"{'Метод':<15} | {'x':<15} | {'Значение'}")
    print("------------------------------------------------")
    print(f"{'Простой поиск':<15} | {sr_point[0]:<7.3f}, {sr_point[1]:<7.3f} | {sr_value:.6f}")
    print(f"{'Алгоритм 1':<15} | {a1_point[0]:<7.3f}, {a1_point[1]:<7.3f} | {a1_value:.6f}")
    print(f"{'Алгоритм 2':<15} | {a2_point[0]:<7.3f}, {a2_point[1]:<7.3f} | {a2_value:.6f}")
    print(f"{'Алгоритм 3':<15} | {a3_point[0]:<7.3f}, {a3_point[1]:<7.3f} | {a3_value:.6f}")
    
    # Подготовка данных для визуализации
    results = [
        ("Простой поиск", sr_point, sr_conv),
        ("Алгоритм 1", a1_point, a1_conv),
        ("Алгоритм 2", a2_point, a2_conv),
        ("Алгоритм 3", a3_point, a3_conv)
    ]
    
    # Визуализация
    visualize_results(results)

if __name__ == "__main__":
    main()
    