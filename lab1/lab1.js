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

// Функция для построения графика зависимости количества итераций от log(eps)
async function plotFunctionCalls(logEpsValues, dichotomyIterations, goldIterations, fiboIterations) {
    const plot = gnuplot();
    plot.set("title", `"Зависимость количества итераций от log10(eps)"`);
    plot.set("xlabel", '"log10(eps)"');
    plot.set("ylabel", '"Количество итераций"');
    plot.set("grid");

    // Подготовка данных для графика
    const data = logEpsValues.map((logEps, index) => 
        `${logEps} ${dichotomyIterations[index]} ${goldIterations[index]} ${fiboIterations[index]}`
    );

    plot.set("terminal png"); // Сохраняем график в файл
    plot.set(`output "iterations_vs_log_eps.png"`, err => console.error(err));

    plot.plot(
        `'-' using 1:2 title "Дихотомия" with linespoints lw 2 lc rgb '#FF0000' pt 7 ps 1.5, \
        '-' using 1:3 title "Золотое сечение" with linespoints lw 2 lc rgb '#00FF00' pt 7 ps 1.5, \
        '-' using 1:4 title "Фибоначчи" with linespoints lw 2 lc rgb '#0000FF' pt 7 ps 1.5`
    );

    for (let i = 0; i < 3; i++) {
        for (let x of data) {
            plot.println(x);
        }
        plot.println("e");
    }

    plot.end();
}

// Метод дихотомии
function dichotomy(eps) {
    const delta = eps / 2;
    let ai = a0, bi = b0;
    const maxIterCount = ceil(log((b0 - a0) / eps) / log(2.0));
    let iterCount = 0; // Счётчик итераций
    const data = []; // Данные для Excel

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
    }

    writeToExcel("Дихотомия", eps, data);
    return { iterations: iterCount }; // Возвращаем количество итераций
}

// Метод золотого сечения
function gold(eps) {
    const phi = (sqrt(5) + 1) / 2;
    const phi1 = (3 - sqrt(5)) / 2;
    const phi2 = (sqrt(5) - 1) / 2;

    let ai = a0, bi = b0;
    const maxIterCount = ceil(log((b0 - a0) / eps) / log(phi));
    let iterCount = 0; // Счётчик итераций
    const data = []; // Данные для Excel

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
    }

    writeToExcel("Золотое сечение", eps, data);
    return { iterations: iterCount }; // Возвращаем количество итераций
}

// Метод Фибоначчи
function fibo(eps) {
    function Fn(n) {
        return (((1 + sqrt(5.0)) / 2.0) ** n - ((1.0 - sqrt(5.0)) / 2.0) ** n) / sqrt(5.0);
    }

    let ai = a0, bi = b0;
    const maxIterCount = ceil(log((b0 - a0) / eps) / log((1.0 + sqrt(5.0)) / 2.0));
    let iterCount = 0; // Счётчик итераций
    const data = []; // Данные для Excel

    let x1 = ai + (Fn(maxIterCount) / Fn(maxIterCount + 2)) * (bi - ai);
    let x2 = ai + (Fn(maxIterCount + 1) / Fn(maxIterCount + 2)) * (bi - ai);
    let y1 = targetFunction(x1), y2 = targetFunction(x2);

    while (iterCount < maxIterCount - 1) {
        iterCount++;

        if (y1 < y2) {
            bi = x2;
            x2 = x1;
            y2 = y1;
            x1 = ai + (Fn(maxIterCount - iterCount + 1) / Fn(maxIterCount - iterCount + 3)) * (bi - ai);
            y1 = targetFunction(x1);
        } else {
            ai = x1;
            x1 = x2;
            y1 = y2;
            x2 = ai + (Fn(maxIterCount - iterCount + 2) / Fn(maxIterCount - iterCount + 3)) * (bi - ai);
            y2 = targetFunction(x2);
        }

        printRes("Фибоначчи", iterCount, ai, bi, eps, data);
    }

    writeToExcel("Фибоначчи", eps, data);
    return { iterations: iterCount }; // Возвращаем количество итераций
}

// Основная функция
async function main() {
    const epsilons = [0.1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7];
    const logEpsValues = epsilons.map(eps => Math.log10(eps)); // Логарифмическая шкала для eps

    // Объявление массивов для сбора данных
    const dichotomyIterations = [];
    const goldIterations = [];
    const fiboIterations = [];

    for (const eps of epsilons) {
        const dichotomyData = dichotomy(eps);
        const goldData = gold(eps);
        const fiboData = fibo(eps);

        // Сбор данных о количестве итераций
        dichotomyIterations.push(dichotomyData.iterations);
        goldIterations.push(goldData.iterations);
        fiboIterations.push(fiboData.iterations);
    }

    // Построение графика зависимости количества итераций от log(eps)
    await plotFunctionCalls(logEpsValues, dichotomyIterations, goldIterations, fiboIterations);
}

main();