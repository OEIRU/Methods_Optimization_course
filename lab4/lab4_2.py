import math
import random
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Point:
    """
    Класс для представления точки в двумерном пространстве
    """
    def __init__(self, x=0.0, y=0.0):
        """Инициализация точки с заданными координатами"""
        self.x = x
        self.y = y
        
    def __sub__(self, other):
        """Разность точек (вычитание)"""
        return Point(self.x - other.x, self.y - other.y)
        
    def __add__(self, other):
        """Сумма точек (сложение)"""
        return Point(self.x + other.x, self.y + other.y)
        
    def __mul__(self, scalar):
        """Умножение точки на скаляр"""
        return Point(scalar * self.x, scalar * self.y)
        
    __rmul__ = __mul__  

class Const:
    """
    Класс для представления параметров целевой функции
    
    Атрибуты:
        C (float): коэффициент
        a (float): центр по оси x
        b (float): центр по оси y
    """
    def __init__(self, C, a, b):
        self.C = C  # Высота функции
        self.a = a  # Центр по x
        self.b = b  # Центр по y
        
    @staticmethod
    def get_func(point):
        """
        Вычисление значения целевой функции в заданной точке
        
        Args:
            point (Point): точка в двумерном пространстве
            
        Returns:
            float: значение функции в точке
        """
        func = 0.0
        for c in OptimizationParams.constVec:
            dx = point.x - c.a
            dy = point.y - c.b
            func += c.C / (1.0 + dx*dx + dy*dy)  # Сумма рациональных функций
        return func

class OptimizationParams:
    """
    Класс параметров оптимизации
    
    Атрибуты:
        constVec (list): список параметров функции
        eps (float): точность
        P (float): вероятность
        maxiter (int): максимальное число итераций
        baseN (int): базовое число итераций
        numIterations (int): рассчитанное число итераций
    """
    constVec = [  # Параметры для целевой функции
        Const(1, 0, -1),
        Const(2, 0, -4),
        Const(10, 3, -2),
        Const(5, -7, -6),
        Const(7, 6, -10),
        Const(9, 6, 1)
    ]
    eps = 1e-6  # Точность вычислений
    P = 0.999   # Вероятность
    maxiter = 10000  # Максимальное число итераций
    baseN = 400      # Базовое число итераций
    # Рассчитанное число итераций
    numIterations = int(math.log(1 - P) / math.log(1 - eps * eps / baseN))

class RandomGenerator:
    """
    Реализация паттерна Singleton для генератора случайных чисел
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Получение единственного экземпляра генератора"""
        if cls._instance is None:
            cls._instance = random.Random(time.time())  # Инициализация с текущим временем
        return cls._instance

def MakeRandomPoint(point, xMin=-10, xMax=10, yMin=-10, yMax=10):
    """
    Генерация случайной точки в заданном диапазоне
    
    Args:
        point (Point): точка для обновления
        xMin (float): минимальное значение x
        xMax (float): максимальное значение x
        yMin (float): минимальное значение y
        yMax (float): максимальное значение y
    """
    rng = RandomGenerator.get_instance()
    point.x = rng.uniform(xMin, xMax)
    point.y = rng.uniform(yMin, yMax)

def GetFunc(point):
    """
    Вычисление значения целевой функции
    
    Args:
        point (Point): точка в двумерном пространстве
        
    Returns:
        float: значение функции в точке
    """
    return Const.get_func(point)

def SimpleRandomSearch():
    """
    Простой случайный поиск с фиксацией лучшего результата
    
    Returns:
        dict: результат поиска (точка, значение, текстовое описание)
    """
    currentPoint = Point()
    maxPoint = Point()
    MakeRandomPoint(maxPoint, -10, 10, -10, 10)
    maxFunc = GetFunc(maxPoint)
    
    for _ in range(OptimizationParams.numIterations):
        MakeRandomPoint(currentPoint, -10, 10, -10, 10)
        currentFunc = GetFunc(currentPoint)
        if currentFunc > maxFunc:
            maxFunc = currentFunc
            maxPoint = currentPoint
            
    return {
        'point': maxPoint,
        'value': maxFunc,
        'text': (f"N= {OptimizationParams.numIterations}\n"
                f"(x, y) = ({maxPoint.x:.6f}, {maxPoint.y:.6f})\n"
                f"f= {maxFunc:.6f}")
    }

def HyperSquareSearch(startPoint):
    """
    Алгоритм гиперквадратного поиска с адаптивным изменением области
    
    Args:
        startPoint (Point): начальная точка
        
    Returns:
        dict: результат поиска (точка, значение, количество вызовов функции)
    """
    # Инициализация параметров поиска
    xMin = max(-10.0, startPoint.x - 5)
    xMax = min(10.0, startPoint.x + 5)
    yMin = max(-10.0, startPoint.y - 5)
    yMax = min(10.0, startPoint.y + 5)
    numPoints = 100
    alpha = 2.0
    points = [Point() for _ in range(numPoints)]
    maxFunc = GetFunc(startPoint)
    funcCalls = 1
    maxPoint = Point(startPoint.x, startPoint.y)
    iteration = 0
    prevMaxFunc = maxFunc - 1e6
    
    # Основной цикл поиска
    while abs(prevMaxFunc - maxFunc) > OptimizationParams.eps and iteration < OptimizationParams.maxiter:
        prevMaxFunc = maxFunc
        iteration += 1
        
        # Генерация новых точек
        for p in points:
            MakeRandomPoint(p, xMin, xMax, yMin, yMax)
            
        # Поиск максимума среди новых точек
        for p in points:
            currentFunc = GetFunc(p)
            funcCalls += 1
            if currentFunc > maxFunc:
                maxFunc = currentFunc
                maxPoint = Point(p.x, p.y)
                
        # Сужение области поиска
        newWidthX = (xMax - xMin) / (2 * alpha)
        newWidthY = (yMax - yMin) / (2 * alpha)
        xMin = max(-10.0, maxPoint.x - newWidthX)
        xMax = min(10.0, maxPoint.x + newWidthX)
        yMin = max(-10.0, maxPoint.y - newWidthY)
        yMax = min(10.0, maxPoint.y + newWidthY)
        
    return {
        'point': maxPoint,
        'value': maxFunc,
        'calls': funcCalls
    }

def LinearSearch():
    """
    Линейный поиск с использованием случайных направлений
    
    Returns:
        dict: результат поиска (точка, значение, текстовое описание)
    """
    startPoint = Point()
    MakeRandomPoint(startPoint, -10, 10, -10, 10)
    maxFunc = GetFunc(startPoint)
    maxPoint = Point(startPoint.x, startPoint.y)
    funcCalls = 1
    directions = [Point() for _ in range(5)]
    
    # Генерация случайных направлений
    for d in directions:
        MakeRandomPoint(d, -1, 1, -1, 1)
        
    # Поиск по каждому направлению
    for direction in directions:
        stepSize = 0.5
        currentPoint = Point(maxPoint.x, maxPoint.y)
        steps = 0
        
        while steps < 50:
            currentPoint = currentPoint + direction * stepSize
            # Проверка выхода за границы
            if not (-10 <= currentPoint.x <= 10 and -10 <= currentPoint.y <= 10):
                break
                
            currentFunc = GetFunc(currentPoint)
            funcCalls += 1
            
            if currentFunc > maxFunc:
                maxFunc = currentFunc
                maxPoint = Point(currentPoint.x, currentPoint.y)
                stepSize *= 1.1  # Увеличение шага при успехе
            else:
                stepSize *= 0.9  # Уменьшение шага при неудаче
                if stepSize < 0.001:
                    break
            steps += 1
            
    return {
        'point': maxPoint,
        'value': maxFunc,
        'text': (f"Линейный поиск:\n"
                f"(x, y) = ({maxPoint.x:.6f}, {maxPoint.y:.6f})\n"
                f"f= {maxFunc:.6f}\n"
                f"Вызовов функции: {funcCalls}")
    }

def Algorithm1(restartCount):
    """
    Алгоритм 1: Многократный гиперквадратный поиск
    
    Args:
        restartCount (int): количество перезапусков
        
    Returns:
        dict: результат поиска (точка, значение, текстовое описание)
    """
    startPoint = Point()
    MakeRandomPoint(startPoint, -10, 10, -10, 10)
    result = HyperSquareSearch(startPoint)
    maxFunc = result['value']
    maxPoint = result['point']
    totalFuncCalls = result['calls']
    successCount = 0
    
    while successCount < restartCount:
        tempPoint = Point()
        MakeRandomPoint(tempPoint, -10, 10, -10, 10)
        result = HyperSquareSearch(tempPoint)
        totalFuncCalls += result['calls']
        
        if result['value'] > maxFunc:
            maxFunc = result['value']
            maxPoint = result['point']
            successCount = 0
        else:
            successCount += 1
            
    return {
        'point': maxPoint,
        'value': maxFunc,
        'text': (f"\nАлгоритм 1: Многократный гиперквадратный поиск\n"
                f"Общее количество вызовов функции: {totalFuncCalls}\n"
                f"(x, y) = ({maxPoint.x:.6f}, {maxPoint.y:.6f})\n"
                f"Значение функции = {maxFunc:.6f}")
    }

def Algorithm2(attemptCount):
    """
    Алгоритм 2: Локальный поиск с перезапусками
    
    Args:
        attemptCount (int): количество попыток перезапуска
        
    Returns:
        dict: результат поиска (точка, значение, текстовое описание)
    """
    localPoint = Point()
    MakeRandomPoint(localPoint, -10, 10, -10, 10)
    result = HyperSquareSearch(localPoint)
    maxFunc = result['value']
    maxPoint = result['point']
    totalFuncCalls = result['calls']
    
    while True:
        currentPoint = Point(maxPoint.x, maxPoint.y)
        result = HyperSquareSearch(currentPoint)
        totalFuncCalls += result['calls']
        
        successCount = 0
        while True:
            MakeRandomPoint(currentPoint, -10, 10, -10, 10)
            currentFunc = GetFunc(currentPoint)
            totalFuncCalls += 1
            successCount += 1
            
            if currentFunc > maxFunc or successCount >= attemptCount:
                break
                
        if currentFunc > maxFunc:
            maxFunc = currentFunc
            maxPoint = currentPoint
        else:
            break
            
    return {
        'point': maxPoint,
        'value': maxFunc,
        'text': (f"\nАлгоритм 2: Локальный поиск с перезапусками\n"
                f"Общее количество вызовов функции: {totalFuncCalls}\n"
                f"(x, y) = ({maxPoint.x:.6f}, {maxPoint.y:.6f})\n"
                f"Значение функции = {maxFunc:.6f}")
    }

def Algorithm3(attemptCount):
    """
    Алгоритм 3: Стратегия с экстраполяцией
    
    Args:
        attemptCount (int): количество попыток перезапуска
        
    Returns:
        dict: результат поиска (точка, значение, текстовое описание)
    """
    lambda_ = 2.0
    currentPoint = Point()
    MakeRandomPoint(currentPoint, -10, 10, -10, 10)
    result = HyperSquareSearch(currentPoint)
    maxFunc = result['value']
    maxPoint = result['point']
    totalFuncCalls = result['calls']
    continueSearch = True
    
    while continueSearch:
        direction = maxPoint - currentPoint
        tempPoint = Point(maxPoint.x, maxPoint.y)
        tempFunc = GetFunc(tempPoint)
        totalFuncCalls += 1
        deltaF = 0
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            tempPoint1 = tempPoint + lambda_ * direction
            # Проверка выхода за границы
            if not (-10 <= tempPoint1.x <= 10 and -10 <= tempPoint1.y <= 10):
                break
                
            tempFunc1 = GetFunc(tempPoint1)
            totalFuncCalls += 1
            deltaF = tempFunc1 - tempFunc
            
            if deltaF <= 0:
                break
                
            tempPoint = tempPoint1
            tempFunc = tempFunc1
            steps += 1
            
        if deltaF > 0:
            result = HyperSquareSearch(tempPoint)
            totalFuncCalls += result['calls']
            
            if result['value'] - maxFunc > OptimizationParams.eps:
                maxFunc = result['value']
                maxPoint = result['point']
                continueSearch = True
            else:
                found = False
                attempt = 0
                
                while not found and attempt < attemptCount:
                    direction = Point()
                    MakeRandomPoint(direction, -1, 1, -1, 1)
                    tempPoint = Point(maxPoint.x, maxPoint.y)
                    attempt += 1
                    steps = 0
                    
                    while steps < max_steps:
                        tempPoint1 = tempPoint + lambda_ * direction
                        if not (-10 <= tempPoint1.x <= 10 and -10 <= tempPoint1.y <= 10):
                            break
                            
                        tempFunc1 = GetFunc(tempPoint1)
                        totalFuncCalls += 1
                        deltaF = tempFunc1 - tempFunc
                        
                        if deltaF <= 0:
                            break
                            
                        tempPoint = tempPoint1
                        tempFunc = tempFunc1
                        steps += 1
                        
                    if deltaF > 0:
                        result = HyperSquareSearch(tempPoint)
                        totalFuncCalls += result['calls']
                        found = (result['value'] - maxFunc > OptimizationParams.eps)
                        
                    if found:
                        maxFunc = result['value']
                        maxPoint = result['point']
                        break
                        
                if not found:
                    continueSearch = False
        else:
            found = False
            attempt = 0
            
            while not found and attempt < attemptCount:
                direction = Point()
                MakeRandomPoint(direction, -1, 1, -1, 1)
                tempPoint = Point(maxPoint.x, maxPoint.y)
                attempt += 1
                steps = 0
                
                while steps < max_steps:
                    tempPoint1 = tempPoint + lambda_ * direction
                    if not (-10 <= tempPoint1.x <= 10 and -10 <= tempPoint1.y <= 10):
                        break
                        
                    tempFunc1 = GetFunc(tempPoint1)
                    totalFuncCalls += 1
                    deltaF = tempFunc1 - tempFunc
                    
                    if deltaF <= 0:
                        break
                        
                    tempPoint = tempPoint1
                    tempFunc = tempFunc1
                    steps += 1
                    
                if deltaF > 0:
                    result = HyperSquareSearch(tempPoint)
                    totalFuncCalls += result['calls']
                    found = (result['value'] - maxFunc > OptimizationParams.eps)
                    
                if found:
                    maxFunc = result['value']
                    maxPoint = result['point']
                else:
                    continueSearch = False
                    
    return {
        'point': maxPoint,
        'value': maxFunc,
        'text': (f"\nАлгоритм 3: Стратегия с экстраполяцией\n"
                f"Общее количество вызовов функции: {totalFuncCalls}\n"
                f"(x, y) = ({maxPoint.x:.6f}, {maxPoint.y:.6f})\n"
                f"Значение функции = {maxFunc:.6f}")
    }

def plot_function_and_results(ax, results=None):
    """
    Построение графика целевой функции с результатами оптимизации
    
    Args:
        ax (matplotlib.axes.Axes): ось для построения графика
        results (list): список результатов оптимизации
    """
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Вычисление значений функции для графика
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = Point(X[i,j], Y[i,j])
            Z[i,j] = Const.get_func(p)
            
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    
    # Отображение результатов оптимизации
    if results:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (name, point, value) in enumerate(results):
            ax.scatter(point.x, point.y, color=colors[idx % len(colors)], 
                      label=f'{name}: ({point.x:.2f}, {point.y:.2f})', s=100)
                      
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Function Contour Plot with Optimization Results')
    ax.legend(loc='upper right')

def create_gui():
    """
    Создание графического интерфейса пользователя
    """
    root = tk.Tk()
    root.title("Оптимизация функции")
    
    # Переменные для параметров оптимизации
    eps_var = tk.DoubleVar(value=OptimizationParams.eps)
    count_var = tk.IntVar(value=40)
    P_var = tk.DoubleVar(value=OptimizationParams.P)
    
    # Поля ввода параметров
    ttk.Label(root, text="eps:").grid(row=0, column=0)
    ttk.Entry(root, textvariable=eps_var).grid(row=0, column=1)
    ttk.Label(root, text="count:").grid(row=1, column=0)
    ttk.Entry(root, textvariable=count_var).grid(row=1, column=1)
    ttk.Label(root, text="P:").grid(row=2, column=0)
    ttk.Entry(root, textvariable=P_var).grid(row=2, column=1)
    
    # Область вывода результатов
    result_text = tk.Text(root, height=15, width=60)
    result_text.grid(row=4, column=0, columnspan=4, padx=5, pady=5)
    
    # Область для графика
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=5, column=0, columnspan=4)
    
    results_history = []  # История результатов для отображения на графике
    
    def run_algorithm(alg_func):
        """
        Запуск алгоритма оптимизации
        
        Args:
            alg_func (function): функция алгоритма оптимизации
        """
        # Обновление параметров из GUI
        OptimizationParams.eps = eps_var.get()
        OptimizationParams.P = P_var.get()
        OptimizationParams.numIterations = int(math.log(1 - OptimizationParams.P) / 
                                              math.log(1 - OptimizationParams.eps * OptimizationParams.eps / 
                                                       OptimizationParams.baseN))
        
        result_text.delete(1.0, tk.END)  # Очистка предыдущих результатов
        result = alg_func(count_var.get())  # Запуск алгоритма
        
        if isinstance(result, dict):
            result_text.insert(tk.END, result['text'])  # Вывод результатов
            # Сохранение результата для графика
            alg_name = alg_func.__name__
            results_history.append((alg_name, result['point'], result['value']))
        else:
            result_text.insert(tk.END, result)
            
        # Обновление графика
        ax.clear()
        plot_function_and_results(ax, results_history[-5:])  # Отображение последних 5 результатов
        canvas.draw()
    
    # Кнопки для запуска алгоритмов
    ttk.Button(root, text="Алгоритм 1", command=lambda: run_algorithm(lambda cnt: Algorithm1(cnt))).grid(row=3, column=0)
    ttk.Button(root, text="Алгоритм 2", command=lambda: run_algorithm(lambda cnt: Algorithm2(cnt))).grid(row=3, column=1)
    ttk.Button(root, text="Алгоритм 3", command=lambda: run_algorithm(lambda cnt: Algorithm3(cnt))).grid(row=3, column=2)
    ttk.Button(root, text="Линейный поиск", command=lambda: run_algorithm(lambda _: LinearSearch())).grid(row=3, column=3)
    
    # Инициализация графика
    plot_function_and_results(ax)
    canvas.draw()
    root.mainloop()

if __name__ == "__main__":
    create_gui()