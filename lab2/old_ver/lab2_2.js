// Функция для одномерного поиска по направлению
function lineSearch(f, x, direction) {
    const alpha = 0.1; // Начальное значение шага
    const beta = 0.5;  // Коэффициент уменьшения шага
    const maxIterations = 100;
    let alphaStar = alpha;

    for (let i = 0; i < maxIterations; i++) {
        const xNew = x.map((val, idx) => val + alphaStar * direction[idx]);
        if (f(xNew) < f(x)) {
            return alphaStar;
        }
        alphaStar *= beta;
    }

    return alphaStar;
}

// Метод наискорейшего спуска (градиентный спуск)
function gradientDescent(f, gradF, x0, epsilon, maxIterations) {
    let x = x0.slice();
    let iter = 0;
    let funcCalls = 0;

    while (iter < maxIterations) {
        const gradient = gradF(x);
        funcCalls++;
        const norm = Math.sqrt(gradient.reduce((sum, val) => sum + val * val, 0));

        if (norm < epsilon) {
            break;
        }

        const direction = gradient.map(val => -val);

        const alphaStar = lineSearch(f, x, direction);
        funcCalls += 10; // Примерное количество вызовов функции в lineSearch

        for (let i = 0; i < x.length; i++) {
            x[i] += alphaStar * direction[i];
        }

        iter++;
    }

    return { x, iter, funcCalls };
}

// Метод Пирсона (сопряжённых градиентов)
function conjugateGradient(f, gradF, x0, epsilon, maxIterations) {
    let x = x0.slice();
    let gradient = gradF(x);
    let direction = gradient.map(val => -val);
    let iter = 0;
    let funcCalls = 1;

    while (iter < maxIterations) {
        const alphaStar = lineSearch(f, x, direction);
        funcCalls += 10; // Примерное количество вызовов функции в lineSearch

        for (let i = 0; i < x.length; i++) {
            x[i] += alphaStar * direction[i];
        }

        const newGradient = gradF(x);
        funcCalls++;
        const beta = (newGradient.reduce((sum, val) => sum + val * val, 0)) / 
                      (gradient.reduce((sum, val) => sum + val * val, 0));

        direction = newGradient.map((val, idx) => -val + beta * direction[idx]);

        gradient = newGradient;

        if (Math.sqrt(gradient.reduce((sum, val) => sum + val * val, 0)) < epsilon) {
            break;
        }

        iter++;
    }

    return { x, iter, funcCalls };
}

// Функция Розенброка и её градиент
function rosenbrock(x) {
    return 100 * Math.pow(x[1] - x[0] * x[0], 2) + Math.pow(1 - x[0], 2);
}

function gradRosenbrock(x) {
    return [
        -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] * x[0])
    ];
}

// Квадратичная функция и её градиент
function quadratic(x) {
    return 100 * Math.pow(x[1] - x[0], 2) + Math.pow(1 - x[0], 2);
}

function gradQuadratic(x) {
    return [
        -200 * (x[1] - x[0]) - 2 * (1 - x[0]),
        200 * (x[1] - x[0])
    ];
}

// Тестовая функция и её градиент
function testFunction(x) {
    const term1 = 3 * Math.exp(-Math.pow(x[0] - 2, 2) - Math.pow((x[1] - 3) / 2, 2));
    const term2 = Math.exp(-Math.pow((x[0] - 1) / 2, 2) - Math.pow(x[1] - 1, 2));
    return term1 + term2;
}

function gradTestFunction(x) {
    const dx1 = 3 * (-2 * (x[0] - 2)) * Math.exp(-Math.pow(x[0] - 2, 2) - Math.pow((x[1] - 3) / 2, 2)) +
                (-2 * (x[0] - 1) / 4) * Math.exp(-Math.pow((x[0] - 1) / 2, 2) - Math.pow(x[1] - 1, 2));
    const dx2 = 3 * (-2 * (x[1] - 3) / 4) * Math.exp(-Math.pow(x[0] - 2, 2) - Math.pow((x[1] - 3) / 2, 2)) +
                (-2 * (x[1] - 1)) * Math.exp(-Math.pow((x[0] - 1) / 2, 2) - Math.pow(x[1] - 1, 2));
    return [dx1, dx2];
}

// Параметры
const epsilon = 1e-6;
const maxIterations = 10000;

// Начальные точки
const startPoints = [
    [0, 0],
    [-1, -1],
    [2, 2]
];

// Запуск исследований
for (const startPoint of startPoints) {
    console.log(`Starting point: [${startPoint}]`);

    // Функция Розенброка
    let result = gradientDescent(rosenbrock, gradRosenbrock, startPoint, epsilon, maxIterations);
    console.log("Gradient Descent (Rosenbrock):", result);

    result = conjugateGradient(rosenbrock, gradRosenbrock, startPoint, epsilon, maxIterations);
    console.log("Conjugate Gradient (Rosenbrock):", result);

    // Квадратичная функция
    result = gradientDescent(quadratic, gradQuadratic, startPoint, epsilon, maxIterations);
    console.log("Gradient Descent (Quadratic):", result);

    result = conjugateGradient(quadratic, gradQuadratic, startPoint, epsilon, maxIterations);
    console.log("Conjugate Gradient (Quadratic):", result);

    // Тестовая функция
    result = gradientDescent(testFunction, gradTestFunction, startPoint, epsilon, maxIterations);
    console.log("Gradient Descent (Test Function):", result);

    result = conjugateGradient(testFunction, gradTestFunction, startPoint, epsilon, maxIterations);
    console.log("Conjugate Gradient (Test Function):", result);
}