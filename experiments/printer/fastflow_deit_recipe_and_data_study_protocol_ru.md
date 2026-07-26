# Протокол сравнения DeiT-рецептов и исследования данных

## Что уже известно

Средний test ROC AUC `0.949385` получен новым DeiT-рецептом на
`printer_384_v2_final`. Он корректно сравним с ResNet18, оценённым на тех же
98 test tiles и тех же трёх seed. Однако это число нельзя напрямую называть
приростом относительно исторического DeiT: исторический запуск использовал
другой test, другой split и другую процедуру оценки.

Текущий test уже раскрыт предыдущим исследованием. Поэтому последующие
результаты на нём являются ретроспективными, а не новым независимым
подтверждением.

## Контролируемое сравнение рецептов

Сравниваются две заранее зафиксированные конфигурации:

| Параметр | Legacy DeiT try_1 | Current DeiT |
|---|---:|---:|
| Источник | Git `870073e` | финальный pipeline |
| Epochs | 24 | 40 |
| Learning rate | 1e-4 | 3e-5 |
| Weight decay | 5e-5 | 1e-5 |
| Flow steps | 8 | 8 |
| Hidden ratio | default Anomalib | 0.5 |
| Batch size | 14 | 10 |
| Gradient clipping | нет | 10.0 |
| Early stopping patience | 3 | нет |
| Train augmentation | исходная сильная | нет |

Одинаковыми остаются:

- manifest `printer_split_v2_final.csv`;
- train, normal-loss validation, calibration и test;
- pretrained DeiT backbone и замороженный feature extractor;
- seed, checkpoint selection и inference;
- calibration-only выбор top-k и threshold;
- основная метрика и bootstrap-код.

Исходный legacy top-k не переносится, потому что он был связан со старым test.
Обе модели проходят один общий calibration-протокол.

Ноутбуки:

- `code/FastFlow_printer_deit_legacy_recipe.ipynb`;
- `code/FastFlow_printer_deit_current_recipe.ipynb`.

Первый проход: seed `42`, только `train_calibrate`. После обоих запусков
сравниваются loss, best epoch, top-k curve, calibration ranking и threshold
errors. Дополнительные paired seeds допускаются только при существенной
seed-чувствительности или близкой calibration-разнице и назначаются до
повторного просмотра test.

## Множественное тестирование

Для дальнейших ablation test не используется в цикле выбора. Процесс:

1. До запуска записать гипотезу, единственное изменение и ожидаемый эффект.
2. Обучить baseline и вариант на одинаковых seed и train subset.
3. Выбирать вариант по normal validation и labeled calibration.
4. Ограничить число вариантов заранее; не продолжать поиск после случайного
   хорошего test.
5. После freeze оценить один финальный вариант на новом будущем holdout.

Поправка Holm или bootstrap-интервалы полезны для заранее заданного семейства
гипотез, но не устраняют адаптивный bias, если варианты выбирались после
просмотра одного test. Единственное надёжное подтверждение после такого
поиска — новая неиспользованная производственная сессия.

Из-за малого числа calibration source groups результаты нужно считать по
source-group и показывать разброс по seed. Tile-level размер выборки нельзя
трактовать как число независимых наблюдений.

## Исследование аугментаций

Минимальный набор ablation:

1. Без аугментаций — текущий baseline.
2. Только слабая фотометрия: небольшой brightness/contrast и редкий blur.
3. Консервативная геометрия, только если она физически допустима для камеры и
   детали.

Сильный `RandomResizedCrop`, `RandomErasing`, частый blur и произвольные
повороты проверяются только как legacy-контроль. Они могут превращать normal
training sample в искусственный outlier или удалять локальную текстуру,
которая нужна для поверхностных дефектов.

Для каждого варианта сохраняются exact transform config, несколько
воспроизводимых preview-изображений, train/normal-val loss, best epoch,
calibration score distributions, top-k sweep и метрики по source groups.

## Исследование объёма и состава train

Нужно разделить два эффекта:

- число уникальных source/object groups;
- число похожих tiles из одного дня или одной детали.

Рекомендуемая последовательность:

1. Зафиксировать вложенные train subsets, например 250, 500, 1000 и все
   доступные normal tiles.
2. Сначала выбирать по source/object groups, затем tiles внутри группы.
3. Сравнить равномерный лимит на дату с использованием всех данных.
4. Не менять calibration/test и preprocessing между вариантами.
5. Отдельно показать число tiles, objects, source images и дат для каждого
   subset.

Чтобы объём данных не смешивался с числом optimizer updates, основной анализ
должен либо фиксировать число шагов оптимизации, либо явно проводить два
контроля: одинаковые epochs и одинаковые steps. Иначе улучшение нельзя
однозначно приписать разнообразию данных.

## Критерии решения

Улучшение считается устойчивым, если направление эффекта сохраняется на
paired seeds и на source-group метриках, а не только на отдельных tiles.
Помимо ROC AUC обязательно сравниваются FPR/FNR при calibration threshold,
object/source errors, loss dynamics и доля независимых source groups.

Окончательный production-вывод требует нового holdout из будущих съёмок.
