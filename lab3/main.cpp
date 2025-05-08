#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>  // Для std::sort
#include <fstream>

// Целевая функция с штрафом
double objectiveFunction(double x, double y, double sigma) {
    double f = 5 * pow(x - y, 2) + pow(x - 2, 2);
    double penalty = std::max(0.0, x + y - 1);
    return f + sigma * pow(penalty, 2);
}

// Метод Нелдера-Мида
std::pair<double, double> nelderMead(double (*func)(double, double, double),
                                     double x0, double y0,
                                     double dx, double dy,
                                     double epsilon,
                                     double sigma) {
    std::vector<std::pair<double, double>> simplex = {
        {x0, y0},
        {x0 + dx, y0},
        {x0, y0 + dy}
    };

    double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma_shrink = 0.5;

    for (int iter = 0; iter < 1000; ++iter) {
        std::vector<double> values;
        for (const auto& point : simplex)
            values.push_back(func(point.first, point.second, sigma));

        std::vector<int> indices = {0, 1, 2};
        std::sort(indices.begin(), indices.end(),
                  [&](int i, int j) { return values[i] < values[j]; });

        int best = indices[0], secondWorst = indices[1], worst = indices[2];

        double centroidX = (simplex[best].first + simplex[secondWorst].first) / 2.0;
        double centroidY = (simplex[best].second + simplex[secondWorst].second) / 2.0;

        // Отражение
        double rx = centroidX + alpha * (centroidX - simplex[worst].first);
        double ry = centroidY + alpha * (centroidY - simplex[worst].second);
        double rValue = func(rx, ry, sigma);

        if (rValue < values[best]) {
            // Расширение
            double ex = centroidX + gamma * (rx - centroidX);
            double ey = centroidY + gamma * (ry - centroidY);
            double eValue = func(ex, ey, sigma);
            if (eValue < rValue)
                simplex[worst] = {ex, ey};
            else
                simplex[worst] = {rx, ry};
        } else if (rValue < values[secondWorst]) {
            simplex[worst] = {rx, ry};
        } else {
            // Сжатие
            double cx = centroidX + rho * (simplex[worst].first - centroidX);
            double cy = centroidY + rho * (simplex[worst].second - centroidY);
            double cValue = func(cx, cy, sigma);
            if (cValue < values[worst])
                simplex[worst] = {cx, cy};
            else {
                for (int i = 1; i < 3; ++i) {
                    simplex[i].first = simplex[best].first + sigma_shrink * (simplex[i].first - simplex[best].first);
                    simplex[i].second = simplex[best].second + sigma_shrink * (simplex[i].second - simplex[best].second);
                }
            }
        }

        // Проверка сходимости
        double maxDist = 0.0;
        for (const auto& point : simplex) {
            double dist = sqrt(pow(point.first - simplex[best].first, 2) +
                               pow(point.second - simplex[best].second, 2));
            maxDist = std::max(maxDist, dist);
        }
        if (maxDist < epsilon) break;
    }

    double bestValue = INFINITY;
    std::pair<double, double> bestPoint;
    for (const auto& point : simplex) {
        double val = func(point.first, point.second, sigma);
        if (val < bestValue) {
            bestValue = val;
            bestPoint = point;
        }
    }
    return bestPoint;
}

// Генерация файла данных и скрипта для gnuplot
void generateGnuplotScript(double xOpt, double yOpt) {
    std::ofstream dataFile("data.txt");
    dataFile << xOpt << " " << yOpt << "\n";
    dataFile.close();

    std::ofstream scriptFile("plot_script.gp");
    scriptFile << "set view 60, 30\n";
    scriptFile << "set xlabel 'x'\n";
    scriptFile << "set ylabel 'y'\n";
    scriptFile << "set zlabel 'f(x, y)'\n";
    scriptFile << "set title 'Целевая функция и ограничение x + y <= 1'\n";
    scriptFile << "set contour base\n";
    scriptFile << "set cntrparam levels 15\n";
    scriptFile << "unset surface\n";
    scriptFile << "set style data lines\n";
    scriptFile << "splot [x=-1:3] [y=-1:3] 5*(x-y)**2 + (x-2)**2, \\\n";
    scriptFile << "     '' using 1:2:(0) with points pt 7 ps 2 lc 'red', \\\n";
    scriptFile << "     '' using 1:(1 - $1):(0) with lines lt 2 lc 'blue'\n";
    scriptFile << "pause -1\n";
    scriptFile.close();
}

void generateGnuplotScript(const std::vector<std::pair<double, double>>& trajectory) {
    // Сохраняем траекторию в файл
    std::ofstream trajFile("trajectory.txt");
    for (const auto& point : trajectory)
        trajFile << point.first << " " << point.second << "\n";
    trajFile.close();

    // Генерируем скрипт gnuplot
    std::ofstream scriptFile("plot_script.gp");
    scriptFile << "set view 60, 30\n";
    scriptFile << "set xlabel 'x'\n";
    scriptFile << "set ylabel 'y'\n";
    scriptFile << "set zlabel 'f(x, y)'\n";
    scriptFile << "set title 'Целевая функция и ограничение x + y <= 1'\n";
    scriptFile << "set pm3d depthorder hidden3d\n";  // Для цветной поверхности
    scriptFile << "set palette defined (0 'blue', 1 'green', 2 'yellow', 3 'red')\n";
    scriptFile << "unset key\n";

    // Рисуем поверхность целевой функции
    scriptFile << "splot [x=-1:3] [y=-1:3] 5*(x-y)**2 + (x-2)**2 with pm3d,\n";

    // Рисуем сетку ограничения x + y = 1
    scriptFile << "     '' using 1:(1 - $1):(0) with lines lc 'white' lw 2 dt 2,\n";

    // Рисуем траекторию оптимизации
    scriptFile << "     'trajectory.txt' using 1:2:(0) with lines lc 'lime' lw 2,\n";

    // Рисуем точку минимума
    scriptFile << "     'data.txt' using 1:2:(0) with points pt 7 ps 2 lc 'red'\n";

    scriptFile << "pause -1\n";
    scriptFile.close();
}

int main() {
    double x0 = 0.0, y0 = 0.0, dx = 0.1, dy = 0.1, epsilon = 1e-6;
    double sigma = 1.0, sigmaMultiplier = 10.0;
    int maxPenaltySteps = 5;

    std::pair<double, double> result;
    std::vector<std::pair<double, double>> trajectory;

    for (int step = 0; step < maxPenaltySteps; ++step) {
        result = nelderMead(objectiveFunction, x0, y0, dx, dy, epsilon, sigma);
        trajectory.push_back(result);  // Сохраняем текущую точку        
        double x = result.first, y = result.second;
        double constraintViolation = x + y - 1;
        std::cout << "Шаг штрафа " << step + 1 << ", sigma = " << sigma << "\n";
        std::cout << "x = " << x << ", y = " << y << "\n";
        std::cout << "Нарушение ограничения: " << constraintViolation << "\n";
        std::cout << "Целевая функция: " << 5 * pow(x - y, 2) + pow(x - 2, 2) << "\n";
        sigma *= sigmaMultiplier;
    }

    generateGnuplotScript(trajectory);

    //generateGnuplotScript(result.first, result.second);

    std::cout << "\nГрафик сохранён в файлы:\n";
    std::cout << "- data.txt\n";
    std::cout << "- plot_script.gp\n";
    std::cout << "Запустите команду для построения графика:\n";
    std::cout << "gnuplot -persist plot_script.gp\n";

    
    return 0;
}