# Скрипт Gnuplot для визуализации
reset session

# Настройки общие
set terminal pngcairo enhanced font 'Verdana,10' size 800,600
set datafile separator ','  # Указываем разделитель
set key outside

# График 1: Сравнение итераций и вызовов функций
set output 'comparison.png'

# Подграфик 1: Итерации
set multiplot layout 1,2

# Итерации
set title "Iterations Comparison"
set xlabel "Start Point"
set ylabel "Iterations"
set style data histograms
set style histogram rowstacked
set boxwidth 0.8
set style fill solid border -1
unset key

plot 'results.csv' using 4:xtic(1) every ::1::1000000:1 notitle with boxes

# Подграфик 2: Вызовы функций
set origin 0.5,0
set size 0.5,1
set title "Function Calls Comparison"
set ylabel "Function Calls"
unset style data
unset style histogram
set style fill solid

plot 'results.csv' using 5:xtic(1) every ::1::1000000:1 notitle with boxes

unset multiplot

# График 2: Сходимость по ошибке
set output 'convergence.png'
set xlabel "Iteration"
set ylabel "Error"
set key outside

plot \
    'history_GD_Rosenbrock_0,0.csv' using 1:2 with lines title 'GD (Rosenbrock)', \
    'history_CG_Rosenbrock_0,0.csv' using 1:2 with lines title 'CG (Rosenbrock)', \
    'history_GD_Quadratic_0,0.csv' using 1:2 with lines title 'GD (Quadratic)', \
    'history_CG_Quadratic_0,0.csv' using 1:2 with lines title 'CG (Quadratic)', \
    

# График 3: Вызовы функций по итерациям
set output 'func_calls.png'
set ylabel "Function Calls"
plot \
    'history_GD_Rosenbrock_0,0.csv' using 1:3 with lines title 'GD (Rosenbrock)', \
    'history_CG_Rosenbrock_0,0.csv' using 1:3 with lines title 'CG (Rosenbrock)', \
    'history_GD_Quadratic_0,0.csv' using 1:3 with lines title 'GD (Quadratic)', \
    'history_CG_Quadratic_0,0.csv' using 1:3 with lines title 'CG (Quadratic)', \
    'history_GD_Test_0,0.csv' using 1:3 with lines title 'GD (Func)', \
    'history_CG_Test_0,0.csv' using 1:3 with lines title 'CG (Func)'