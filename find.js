// Глобальные переменные
const targetFunction = (x) => (x - 5) ** 2;

// Функция для поиска интервала, содержащего минимум функции
function findIntervalWithMin(x0, delta) {
    let xk = x0; // Текущая точка
    let h = delta; // Начальный шаг
    let k = 0; // Счётчик итераций

    // Вывод заголовка таблицы
    console.log(`𝛿 = ${delta}, 𝑥0 = ${x0}`);
    console.log("𝑖\t𝑥𝑖\t𝑓(𝑥𝑖)");

    // Шаг 1: Определяем направление убывания функции
    if (targetFunction(x0) > targetFunction(x0 + delta)) {
        xk = x0 + delta; // Двигаемся вправо
        h = delta;
    } else {
        xk = x0 - delta; // Двигаемся влево
        h = -delta;
    }

    // Массив для хранения точек
    const points = [x0, xk];

    // Вывод начальных значений
    console.log(`${k}\t${x0.toFixed(1)}\t${targetFunction(x0).toFixed(2)}`);
    console.log(`${k + 1}\t${xk.toFixed(1)}\t${targetFunction(xk).toFixed(2)}`);

    // Шаг 2 и 3: Удваиваем шаг и проверяем условие
    while (true) {
        h *= 2; 
        const xNext = xk + h; 

        if (targetFunction(xk) > targetFunction(xNext)) {
            points.push(xNext); 
            k++;
            xk = xNext; 
            console.log(`${k + 1}\t${xk.toFixed(1)}\t${targetFunction(xk).toFixed(2)}`);
        } else {
            console.log(`Интервал, содержащий минимум: [${points[k].toFixed(1)}, ${xNext.toFixed(1)}]`);
            return [points[k], xNext]; 
        }
    }
}

const x0 = 0; 
const delta = 0.1; 

const interval = findIntervalWithMin(x0, delta);
console.log(`Найденный интервал, содержащий минимум: [${interval[0].toFixed(1)}, ${interval[1].toFixed(1)}]`);