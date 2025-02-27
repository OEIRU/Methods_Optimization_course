const { sqrt, log, ceil } = Math;
const XLSX = require("xlsx"); // Для работы с Excel
const gnuplot = require("gnuplot"); // Для построения графиков

// Глобальные переменные
const a0 = -2.0; // Начало отрезка
const b0 = 20.0; // Конец отрезка
const targetFunction = (x) => (x - 5) ** 2; // Целевая функция

// Функция для записи данных в Excel
function writeToExcel(methodName, eps, data) {
    const workbook = XLSX.utils.book_new();
    const worksheet = XLSX.utils.json_to_sheet(data);
    XLSX.utils.book_append_sheet(workbook, worksheet, methodName);
    XLSX.writeFile(workbook, `${methodName}_eps_${eps}.xlsx`);
}

// Функция для вывода результатов
function printRes(name, iter, x1, x2, eps, data) {
    const median = (x1 + x2) / 2.0; // Медиана интервала
    const trueMin = 5.0; // Истинный минимум
    const error = Math.abs(median - trueMin); // Абсолютная ошибка

    console.log("===========================================================");
    console.log(`      Метод: ${name.padEnd(16, " ")}         x1: ${x1.toFixed(15)}`);
    console.log(`   Итерации: ${String(iter).padEnd(16, " ")}         x2: ${x2.toFixed(15)}`);
    console.log(`        eps: ${String(eps.toFixed(7)).padEnd(16, " ")}    Медиана: ${median.toFixed(15)}`);
    console.log(`Абс. ошибка: ${error.toFixed(15)}`);

    // Записываем данные для Excel
    data.push({
        a: x1,
        b: x2,
        "Длина отрезка": Math.abs(x2 - x1),
        "Отношение длин": data.length > 0 ? Math.abs(x2 - x1) / data[data.length - 1]["Длина отрезка"] : 1,
        "Абс. ошибка": error,
    });
}

// Функция для построения графика с тремя линиями
function plotErrors(iterations, dichotomyErrors, goldErrors, fiboErrors, eps) {
    const plot = gnuplot();
    plot.set("title", `Сравнение методов, eps: ${eps}`);
    plot.set("xlabel", "Итерация");
    plot.set("ylabel", "Абсолютная ошибка");
    plot.set("grid");

    // Подготовка данных для графика
    const data = iterations
        .map((iter, index) => `${iter}\t${dichotomyErrors[index]}\t${goldErrors[index]}\t${fiboErrors[index]}`)
        .join("\n");

    console.log("Данные для графика:");
    console.log(data);

    plot.set("terminal", "png"); // Сохраняем график в файл
    plot.set("output", `comparison_eps_${eps}.png`);

    plot.plot(`
        '-' using 1:2 title "Дихотомия" with lines, \
        '-' using 1:3 title "Золотое сечение" with lines, \
        '-' using 1:4 title "Фибоначчи" with lines
    `);

    plot.write(data, (err) => {
        if (err) {
            console.error("Ошибка при построении графика:", err);
        } else {
            console.log(`График сохранён в файл comparison_eps_${eps}.png`);
        }
    });

    plot.end();
}

// Метод дихотомии
function dichotomy(eps) {
    const delta = eps / 2;
    let ai = a0, bi = b0;
    const maxIterCount = ceil(log((b0 - a0) / eps) / log(2.0));
    let iterCount = 0;
    const data = []; // Данные для Excel
    const errors = []; // Абсолютные ошибки для графика
    const iterations = []; // Номера итераций

    while (iterCount < maxIterCount) {
        iterCount++;
        const x1 = (ai + bi - delta) / 2.0;
        const x2 = (ai + bi + delta) / 2.0;

        const y1 = targetFunction(x1);
        const y2 = targetFunction(x2);

        if (y1 < y2) {
            bi = x2;
        } else {
            ai = x1;
        }

        printRes("Дихотомия", iterCount, ai, bi, eps, data);
        errors.push(data[data.length - 1]["Абс. ошибка"]);
        iterations.push(iterCount);
    }

    writeToExcel("Дихотомия", eps, data);
    return { iterations, errors };
}

// Метод золотого сечения
function gold(eps) {
    const phi = (sqrt(5) + 1) / 2;
    const phi1 = (3 - sqrt(5)) / 2;
    const phi2 = (sqrt(5) - 1) / 2;

    let ai = a0, bi = b0;
    const maxIterCount = ceil(log((b0 - a0) / eps) / log(phi));
    let iterCount = 0;
    const data = []; // Данные для Excel
    const errors = []; // Абсолютные ошибки для графика
    const iterations = []; // Номера итераций

    while (iterCount < maxIterCount) {
        iterCount++;
        const x1 = ai + phi1 * (bi - ai);
        const x2 = ai + phi2 * (bi - ai);

        const y1 = targetFunction(x1);
        const y2 = targetFunction(x2);

        if (y1 < y2) {
            bi = x2;
        } else {
            ai = x1;
        }

        printRes("Золотое сечение", iterCount, ai, bi, eps, data);
        errors.push(data[data.length - 1]["Абс. ошибка"]);
        iterations.push(iterCount);
    }

    writeToExcel("Золотое сечение", eps, data);
    return { iterations, errors };
}

// Метод Фибоначчи
function fibo(eps) {
    function fibonacci(n) {
        let a = 0, b = 1;
        for (let i = 0; i < n; i++) {
            [a, b] = [b, a + b];
        }
        return a;
    }

    let ai = a0, bi = b0;
    const maxIterCount = ceil(log((b0 - a0) / eps) / log((1 + sqrt(5)) / 2));
    let iterCount = 0;
    const data = []; // Данные для Excel
    const errors = []; // Абсолютные ошибки для графика
    const iterations = []; // Номера итераций

    const F = Array(maxIterCount + 2).fill(0);
    for (let i = 0; i <= maxIterCount + 1; i++) {
        F[i] = fibonacci(i);
    }

    let x1 = ai + (F[maxIterCount - iterCount - 1] / F[maxIterCount - iterCount + 1]) * (bi - ai);
    let x2 = ai + (F[maxIterCount - iterCount] / F[maxIterCount - iterCount + 1]) * (bi - ai);
    let y1 = targetFunction(x1), y2 = targetFunction(x2);

    while (iterCount < maxIterCount - 1) {
        iterCount++;

        if (y1 < y2) {
            bi = x2;
            x2 = x1;
            y2 = y1;
            x1 = ai + (F[maxIterCount - iterCount - 1] / F[maxIterCount - iterCount + 1]) * (bi - ai);
            y1 = targetFunction(x1);
        } else {
            ai = x1;
            x1 = x2;
            y1 = y2;
            x2 = ai + (F[maxIterCount - iterCount] / F[maxIterCount - iterCount + 1]) * (bi - ai);
            y2 = targetFunction(x2);
        }

        printRes("Фибоначчи", iterCount, ai, bi, eps, data);
        errors.push(data[data.length - 1]["Абс. ошибка"]);
        iterations.push(iterCount);
    }

    iterCount++;
    if (y1 < y2) {
        bi = x2;
    } else {
        ai = x1;
    }

    printRes("Фибоначчи", iterCount, ai, bi, eps, data);
    writeToExcel("Фибоначчи", eps, data);
    return { iterations, errors };
}

// Основная функция
function main() {
    const epsilons = [0.1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7];

    for (const eps of epsilons) {
        const dichotomyData = dichotomy(eps);
        const goldData = gold(eps);
        const fiboData = fibo(eps);

        // Построение графика для текущего eps
        plotErrors(
            dichotomyData.iterations,
            dichotomyData.errors,
            goldData.errors,
            fiboData.errors,
            eps
        );
    }
}

main();