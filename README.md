# chair-passer

# Постановка задачи: 
Необходимо по фотографии с изображением дверного проёма и стула сделать вывод о пролезании стула в дверной проём.

# Установка
- Склонировать/скачать репозиторий на локальную машину.

## Зависимости
- Версия Python 3.7.x
- Модули
  * cv2
  * numpy
  * skiimage
  * matplotlib
  * enum
  * os
  * argparse
  * imageai
  * tensorflow
  * keras


# Запуск
Команда запуска в командной строке OS Windows:

python lab.py

Параметры:

1. -h — справка по параметрам;
2. -s — путь до директории, где лежат входные изображение (по умолчанию равен "DataSetV2");
3. -f — путь до файла с проверочными значениями для входных изображений (по умолчанию равен "test_ans.txt");
4. -m — режим работы программы (по умолчанию равен 2):
  * 1 — режим формирования файла с проверочными значениями: происходит заполнение файла названиями изображений из директории, указанной в параметре -s. Ожидаемый ответ для каждого изображения заполняется вручную после формирования файла;
  * 2 — тестовый режим: каждое изображение из директории, указанной в параметре -s обрабатывается [алгоритмом](#финальный-алгоритм) и полученное значение сравнивается с ожидаемым, взятым из файла, указанным в параметре -f. В качестве результата выдаётся точность алгоритма на датасете;
  * 3 — тестовый режим аналогичен режиме 2, но распознавание стула производится не  [представленным алгоритмом](#финальный-алгоритм), а нейросетью.

# Входные параметры алгоритма
Изображение, содержащее два целевых объекта — стул и дверной проём. Стул должен быть зелёного цвета.

# Выходные параметры алгоритма
0 — стул проходит в дверной проём;
1 — стул не проходит в дверной проём;
2 — фото не удовлетворяет требованиям и/или не удалось выбрать между 0 и 1.
 
# Требования к входному изображению
- Изображение должно являться фотографией, сделанной камерой с разрешением как минимум 13 Мп, при достаточном освещении.
- Фотография должна содержать как минимум стул и как минимум дверной проём.
- Стул должен быть зелёного цвета.
- **Стул**
  * **может обладать произвольным числом ножек произвольной формы**
  * **может иметь произвольную форму и число спинок и сидушек**
- Стул должен располагаться на фотографии на удалении как минимум 200 пикселей от правого и левого края изображения.
- Изображение может содержать объекты, перекрывающие стул, если это не влияет на видимость крайних левых и правых точек стула.
- Изображение может содержать объекты, перекрывающие дверной проём (в т.ч. это может быть стул), если объект не перекрывает значительную часть проёма.
На фотографии обязательно должна присутствовать левая и правая границы дверного проёма.
- **Стул может находиться в дверном проёме, стоять рядом с дверным проёмом, стоять перед дверным проёмом, может перекрывать дверной проём на изображении.**
- Центр (по оси X) изображения должен располагаться внутри дверного проёма.
- Самая отдалённая от плоскости дверного проёма точка стула должна находиться не дальше 40 см от плоскости дверного проёма, так как учёт перспективы не поддерживается в текущей версии.
- Стул должен целиком находиться на изображении.
- Наилучшая точность достигается, если плоскость дверного проёма перпендикулярна вектору взгляда камеры, в противном случае ширина дверного проёма будет вычислена без учёта ракурса съёмки и адекватность результата не гарантируется.
- На фотографии дверной проём должен быть изображён вертикально, с погрешностью 0.3 радиан.
- Изображение не следует дополнительно обрабатывать, если это приводит к ухудшению качества распознавания целевых объектов.
- Косяк дверного проёма быть тёмного цвета.
- Дверной проём должен занимать как минимум 80% от высоты изображения.
 
# Дополнительные ограничения обработки изображения
- Вывод о прохождении стула в дверной проём делается без учёта возможности изменения ориентации стула в пространстве.
- Перспектива не учитывается.
- Ракурс не учитывается.
- Освещённость стула должна быть однородной, без затенения или засвеченных частей стула.

# Хронология реализации

Задача состоит из трёх частей:
1. Нахождение стула на изображении.
2. Нахождение дверного проёма на изображении.
3. Нахождение ширины дверного проёма и стула и ответ на вопрос задачи.
 
## Итерация 0 
Изначально предполагалось использовать этот [датасет](https://github.com/MisterProper9000/chair-passer/tree/develop/DataSet)

Для распознавания стула предполагалось использовать детектор Кэнни, для выделения границ, чтобы потом преобразованием Хафа найти почти вертикальные прямые и найти ширину дверного проёма, и путём выделения экстремальных особых точек контуров попытаться выделить детектором Харриса особые точки, принадлежащие стулу.

Чтобы улучшить качество нахождения контуров на изображении, было принято решение использовать детектор Кэнни с адаптивными параметрами, зависящими от средней интенсивности пикселей на изображении. Сравнение результатов работы детектора Кэнни:

![Unblurred wide, tight, auto_canny](https://sun9-42.userapi.com/c857236/v857236987/159aeb/xgttVmjND4A.jpg)

Как видно выделяется слишком много контуров. Попробуем избежать этого заблюрив исходное изображение:

![Blurred wide, tight, auto_canny](https://sun9-6.userapi.com/c857236/v857236987/159b24/OyAEFaPo7_Y.jpg)

Результат стал немного лучше.

Проверим работу детектора Харриса:

![Harris](https://sun9-33.userapi.com/dJEYqCAnSwI6h11gGfvYbwyrdb8ZJYpJ6bX5DQ/EO-ER7Y9TNY.jpg)
![Harris](https://sun9-20.userapi.com/6JBBaq1elCJemLevaxXH4Dv4usQmxwLB04Ne6A/M1BSSlJmZ4I.jpg)

Как видно, узоры на обоях будут сильно мешать детектору обнаружить стул по особым точкам.

Было принято решение сформировать другой [датасет](https://github.com/MisterProper9000/chair-passer/tree/develop/DataSetV2), на котором было бы меньше посторонних объектов, более приятный стул и более одноцветные стены. 

Результаты работы детектора Кэнни:

![Unblurred wide, tight, auto_canny](https://sun9-22.userapi.com/c857236/v857236987/159b86/xiiEjvwlyXg.jpg)

Результаты работы детектора Харриса на нём:

![Harris](https://sun9-54.userapi.com/bMqgCPXgjVIGxdOHIqAeIIaHIX7Aqx3_PfnThw/drGK-IVCOX0.jpg)
![Harris](https://sun9-50.userapi.com/c857236/v857236987/159cb4/njuImcHI8fs.jpg)

Как видно, детектору Харриса больше нравится правый нижний угол изображения, чем стул.

После этого была предпринята попытка изменить метод распознавания стула.

## Итерация 1

Было решено протестировать распознавание стула с помощью TensorFlow и взять за эталонную точность, которую удастся получить с его помощью.

Код итерации 1 не был представлен в [первой демонстрации алгоритма](https://github.com/MisterProper9000/chair-passer/pull/1), так как использовалась не собственная нейросеть, а результаты, полученные с её помощью использовались лишь для улучшения ориентации, куда движется проект.

Для начала сравним точность распознавания стула в обоих датасетах:
На первом датасете точность распознавания составила 78.5%
![Chair accuracy 1](https://sun9-56.userapi.com/c857236/v857236987/159c7e/9ZsXkQPXy3k.jpg)

На втором датасете точность распознавания составила 73.5%
![Chair accuracy 2](https://sun9-53.userapi.com/c857236/v857236884/152db5/yW7aeta1Ssw.jpg)

Учитывая преимущества второго датасета, разница в 5% была признана несущественной.

После решения первой части задачи, можно было приступить к распознаванию дверного проёма.

Планировалось найти все прямые на изображении, с помошью преобразования Хафа и выбрать среди них только те, которые располагаются вертикально, с погрешностью 0.3 радиан. Параметр, отвечающий за настройку погрешности: DOOR\_LINE\_ANGLE\_EPS. Среди ообранных прямых, было бы достаточно выбрать по одной, ближайшей к центру изображения, прямой с обоих сторон относительно центра изображения и посчитать расстояние между ними. Расстояние считается как среднее арифметическое расстояний между парами вершин пересечения прямых с нижним и с верхним краем изображения.

Результат работы преобразования Хафа:

![Door Hough](https://sun9-44.userapi.com/6xmnkV-kqLuvP8JhGlTFiLp0KkQ-2bO9VIdmxg/ufOn8ytviA8.jpg)

Точность алгоритма на втором датасете составила 74%
![Accuracy1](https://sun9-51.userapi.com/2spkMNkPGmBRvD0KVLpt1WM5dnk30rUUe_cohA/diXCLwee6bQ.jpg)

## Итерация 2
 
 На второй итерации был разработан собственный алгоритм распознавания стула, без использования нейросетей.
 
 Стул ищется с помощью бинаризации изображения: не затемняются только те участки изображение, пиксели которых попадают в диапазон известных оттенков стула, полученный в ходе анализа датасета. Далее находится самый правый и самый левый пиксель незатемнённого участка и вычисляется разность между их координатами по оси Х.
 
 Точность такого алгоритма составляет 94%
 ![Accuracy1](https://sun9-70.userapi.com/c857236/v857236884/152e47/55u8Optoh4E.jpg)
 
 В итоге получен значительный прирост точности по сравнению с первой итерацией.
 
# Финальный алгоритм
- Бинаризировать изображение с целью нахождения дверного проёма
- Преобразованием Хафа найти все прямые и оставить только те, которые располагаются почти вертикально (отклонение в пределах 0.1)
- Из отфильтрованных прямых выбрать две ближайшие к центру изображения прямые (ближайшие слева и справа относительно центра)
- Если хотя бы с одной из сторон относительно центра почти вертикальных прямых нет, значит дверной проём отсутствует, либо не распознан.
- Расстояние между отрезками принимается за ширину двери. Расстояние считается как среднее арифметическое расстояний между парами самый верхних точек прямых и самых нижних.
- Бинаризировать изображение с целью нахождения стула (заранее известен диапазон оттенков стула)
- Нахождение ширины стула, как ширины описывающего его бокса.
- Сравнение полученных чисел.