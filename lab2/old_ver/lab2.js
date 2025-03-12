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
function gradientDescent(f, gradF, x0, alpha, epsilon, maxIterations) {
    let x = x0.slice();
    let iter = 0;

    while (iter < maxIterations) {
        const gradient = gradF(x);
        const norm = Math.sqrt(gradient.reduce((sum, val) => sum + val * val, 0));

        if (norm < epsilon) {
            break;
        }

        const direction = gradient.map(val => -val);

        const alphaStar = lineSearch(f, x, direction);

        for (let i = 0; i < x.length; i++) {
            x[i] += alphaStar * direction[i];
        }

        iter++;
    }

    return x;
}

// Метод Пирсона (сопряжённых градиентов)
function conjugateGradient(f, gradF, x0, epsilon, maxIterations) {
    let x = x0.slice();
    let gradient = gradF(x);
    let direction = gradient.map(val => -val);
    let iter = 0;

    while (iter < maxIterations) {
        const alphaStar = lineSearch(f, x, direction);

        for (let i = 0; i < x.length; i++) {
            x[i] += alphaStar * direction[i];
        }

        const newGradient = gradF(x);
        const beta = (newGradient.reduce((sum, val) => sum + val * val, 0)) / 
                      (gradient.reduce((sum, val) => sum + val * val, 0));

        direction = newGradient.map((val, idx) => -val + beta * direction[idx]);

        gradient = newGradient;

        if (Math.sqrt(gradient.reduce((sum, val) => sum + val * val, 0)) < epsilon) {
            break;
        }

        iter++;
    }

    return x;
}

// Пример функции и её градиента
function f(x) {
    return x[0] * x[0] + x[1] * x[1]; // Пример: f(x, y) = x^2 + y^2
}

function gradF(x) {
    return [2 * x[0], 2 * x[1]]; // Градиент функции f(x, y)
}

// Начальные параметры
const x0 = [1, 1]; // Начальная точка
const alpha = 0.1; // Параметр для градиентного спуска
const epsilon = 1e-6; // Точность
const maxIterations = 1000; // Максимальное число итераций

// Запуск методов
const resultGradientDescent = gradientDescent(f, gradF, x0, alpha, epsilon, maxIterations);
console.log("Gradient Descent Result:", resultGradientDescent);

const resultConjugateGradient = conjugateGradient(f, gradF, x0, epsilon, maxIterations);
console.log("Conjugate Gradient Result:", resultConjugateGradient);

// UNIVERSAL
// f(x) = 100(x_2-x_1)^2 + (1-x_1)^2  
// OUR
// f(x,y) = 3 exp -((x-2))^2 - ((y-3)/2)^2 + exp -((x-1)/2)^2 - ((y-1))^2  

