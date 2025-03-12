const fs = require('fs');

// Функция для одномерного поиска по направлению
function lineSearch(f, x, direction) {
    let alphaStar = 0.1;
    let funcCalls = 0;
    const beta = 0.5;
    const maxIterations = 100;
    const minAlpha = 1e-10;

    for (let i = 0; i < maxIterations && alphaStar > minAlpha; i++) {
        funcCalls++;
        const xNew = x.map((val, idx) => val + alphaStar * direction[idx]);
        if (f(xNew) < f(x)) {
            return { alpha: alphaStar, calls: funcCalls };
        }
        alphaStar *= beta;
    }
    return { alpha: alphaStar, calls: funcCalls };
}

// Метод градиентного спуска (с историей итераций)
function gradientDescent(f, gradF, x0, epsilon, maxIterations) {
    let x = x0.slice();
    let iter = 0;
    let funcCalls = 0;
    const history = [];

    while (iter < maxIterations) {
        const gradient = gradF(x);
        funcCalls++;
        const norm = Math.sqrt(gradient.reduce((sum, val) => sum + val * val, 0));

        if (norm < epsilon) break;

        const direction = gradient.map(val => -val);
        const lsResult = lineSearch(f, x, direction);
        funcCalls += lsResult.calls;
        const alphaStar = lsResult.alpha;

        x = x.map((val, i) => val + alphaStar * direction[i]);

        history.push({
            iteration: iter,
            error: norm,
            funcCalls: funcCalls
        });

        iter++;
    }

    return { x, iter, funcCalls, history };
}

// Метод сопряженных градиентов (с историей итераций)
function conjugateGradient(f, gradF, x0, epsilon, maxIterations) {
    let x = x0.slice();
    let gradient = gradF(x);
    let direction = gradient.map(val => -val);
    let iter = 0;
    let funcCalls = 1;
    const history = [];

    while (iter < maxIterations) {
        const lsResult = lineSearch(f, x, direction);
        const alphaStar = lsResult.alpha;
        funcCalls += lsResult.calls;

        x = x.map((val, i) => val + alphaStar * direction[i]);

        const newGradient = gradF(x);
        funcCalls++;
        const betaNumerator = newGradient.reduce((sum, val) => sum + val * val, 0);
        const betaDenominator = gradient.reduce((sum, val) => sum + val * val, 0);
        const beta = betaDenominator === 0 ? 0 : betaNumerator / betaDenominator;

        direction = newGradient.map((val, idx) => -val + beta * direction[idx]);
        gradient = newGradient;

        const norm = Math.sqrt(gradient.reduce((sum, val) => sum + val * val, 0));
        if (norm < epsilon) break;

        history.push({
            iteration: iter,
            error: norm,
            funcCalls: funcCalls
        });

        iter++;
    }

    return { x, iter, funcCalls, history };
}

// Функции и их градиенты
function rosenbrock(x) {
    return 100 * Math.pow(x[1] - x[0] * x[0], 2) + Math.pow(1 - x[0], 2);
}

function gradRosenbrock(x) {
    return [
        -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] * x[0])
    ];
}

function quadratic(x) {
    return 100 * Math.pow(x[1] - x[0], 2) + Math.pow(1 - x[0], 2);
}

function gradQuadratic(x) {
    return [
        -200 * (x[1] - x[0]) - 2 * (1 - x[0]),
        200 * (x[1] - x[0])
    ];
}

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

// Функции записи
function writeResultsToFile(filename, data) {
    const headers = ['Start Point', 'Method', 'Result', 'Iterations', 'Func Calls'];
    const rows = data.map(item => [
        item.startPoint.join(','),
        item.method,
        item.x.join(','),
        item.iter,
        item.funcCalls
    ]);
    const fileContent = [headers.join(','),
                         ...rows.map(row => row.join(','))].join('\n');
    fs.writeFileSync(filename, fileContent);
}

function writeHistoryToFile(historyData, filename) {
    const headers = ['Iteration', 'Error', 'Func Calls'];
    const rows = historyData.map(entry => [
        entry.iteration,
        entry.error,
        entry.funcCalls
    ]);
    const fileContent = [headers.join(','),
                         ...rows.map(row => row.join(','))].join('\n');
    fs.writeFileSync(filename, fileContent);
}

// Параметры
const epsilon = 1e-6;
const maxIterations = 500;
const startPoints = [
    [0, 0],
    [-1, -1],
    [2, 2]
];

// Сбор данных
const results = [];
const historyData = {};

// Запуск исследований
for (const startPoint of startPoints) {
    console.log(`Starting point: [${startPoint}]`);

    // Функция Розенброка
    const rosenGD = gradientDescent(rosenbrock, gradRosenbrock, startPoint, epsilon, maxIterations);
    results.push({
        startPoint: startPoint,
        method: 'Gradient Descent (Rosenbrock)',
        x: rosenGD.x,
        iter: rosenGD.iter,
        funcCalls: rosenGD.funcCalls
    });
    historyData[`GD_Rosenbrock_${startPoint}`] = rosenGD.history;

    const rosenCG = conjugateGradient(rosenbrock, gradRosenbrock, startPoint, epsilon, maxIterations);
    results.push({
        startPoint: startPoint,
        method: 'Conjugate Gradient (Rosenbrock)',
        x: rosenCG.x,
        iter: rosenCG.iter,
        funcCalls: rosenCG.funcCalls
    });
    historyData[`CG_Rosenbrock_${startPoint}`] = rosenCG.history;

    // Квадратичная функция
    const quadGD = gradientDescent(quadratic, gradQuadratic, startPoint, epsilon, maxIterations);
    results.push({
        startPoint: startPoint,
        method: 'Gradient Descent (Quadratic)',
        x: quadGD.x,
        iter: quadGD.iter,
        funcCalls: quadGD.funcCalls
    });
    historyData[`GD_Quadratic_${startPoint}`] = quadGD.history;

    const quadCG = conjugateGradient(quadratic, gradQuadratic, startPoint, epsilon, maxIterations);
    results.push({
        startPoint: startPoint,
        method: 'Conjugate Gradient (Quadratic)',
        x: quadCG.x,
        iter: quadCG.iter,
        funcCalls: quadCG.funcCalls
    });
    historyData[`CG_Quadratic_${startPoint}`] = quadCG.history;

    // Тестовая функция
    const testGD = gradientDescent(testFunction, gradTestFunction, startPoint, epsilon, maxIterations);
    results.push({
        startPoint: startPoint,
        method: 'Gradient Descent (Test)',
        x: testGD.x,
        iter: testGD.iter,
        funcCalls: testGD.funcCalls
    });
    historyData[`GD_Test_${startPoint}`] = testGD.history;

    const testCG = conjugateGradient(testFunction, gradTestFunction, startPoint, epsilon, maxIterations);
    results.push({
        startPoint: startPoint,
        method: 'Conjugate Gradient (Test)',
        x: testCG.x,
        iter: testCG.iter,
        funcCalls: testCG.funcCalls
    });
    historyData[`CG_Test_${startPoint}`] = testCG.history;
}

// Запись результатов
writeResultsToFile('results.csv', results);

// Запись истории
for (const [key, hist] of Object.entries(historyData)) {
    writeHistoryToFile(hist, `history_${key}.csv`);
}

console.log('Файлы созданы!');