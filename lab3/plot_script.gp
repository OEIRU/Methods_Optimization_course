set view 60, 30
set xlabel 'x'
set ylabel 'y'
set zlabel 'f(x, y)'
set title 'Целевая функция и ограничение x + y <= 1'
set contour base
set cntrparam levels 15
unset surface
set style data lines

# Рисуем целевую функцию и точку минимума
splot [x=-1:3] [y=-1:3] 5*(x-y)**2 + (x-2)**2, \
      'data.txt' using 1:2:(0) with points pt 7 ps 2 lc 'red'

# Добавляем линию ограничения x + y = 1 в виде параметрической функции
set parametric
set urange [0:1]
set vrange [0:1]
splot u, 1 - u, 0 lc 'blue' lw 2 dt 2

pause -1