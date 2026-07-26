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

## Зафиксированный first pass

Аудит доступного normal train после отделения normal validation:

| Дата | Tiles | Source images | Objects |
|---|---:|---:|---:|
| 2024-12-16 | 72 | 5 | 14 |
| 2025-01-15 | 44 | 2 | 44 |
| 2025-01-29 | 28 | 5 | 13 |
| 2025-04-02 | 678 | 75 | 164 |
| 2025-05-26 | 108 | 31 | 78 |
| 2025-05-28 | 741 | 117 | 487 |

Всего доступно 1671 tile, 235 source images и 800 objects. Две крупнейшие
даты содержат 1419/1671 (`84.9%`) tiles. Текущий date-balanced subset из 1000
содержит 748/1000 (`74.8%`) tiles этих дат.

До первого запуска зафиксировано семейство из пяти вариантов на seed 42:

| Config | Единственное изменение относительно current DeiT |
|---|---|
| `deit_data_mild_photo_1000` | слабый ColorJitter и редкий GaussianBlur |
| `deit_data_strong_aug_1000` | исходная сильная legacy augmentation |
| `deit_data_balanced_500` | вложенный date-balanced train из 500 tiles |
| `deit_data_all_1671` | все 1671 допустимых normal tiles |
| `deit_data_object_uniform_1000` | 1000 tiles с object-uniform sampling без балансировки дат |

Baseline — ранее выполненный `deit_base_distilled_384`, seed 42. Проверено,
что его 1000 train paths точно совпадают с
`train_rank_date_balanced <= 1000` нового data-study manifest, а
normal validation, calibration и test rows идентичны. Поэтому новый baseline
не переобучается и один GPU-run не расходуется повторно.

Состав фиксированных subsets:

| Subset | Tiles | Source images | Objects | Tiles по шести датам |
|---|---:|---:|---:|---|
| date-balanced 500 | 500 | 167 | 397 | 72/44/28/124/108/124 |
| date-balanced 1000 | 1000 | 212 | 687 | 72/44/28/374/108/374 |
| all 1671 | 1671 | 235 | 800 | 72/44/28/678/108/741 |
| object-uniform 1000 | 1000 | 235 | 800 | 24/44/17/249/83/583 |

Основной first-pass критерий — calibration source-group-balanced tile ROC
AUC. Дополнительно проверяются object/source ROC AUC, FPR/FNR при
calibration threshold, форма top-k curve, best epoch и loss dynamics.

В следующий этап проходит не более одного варианта. Gate:

- технически валидное обучение и конечные loss;
- улучшение основной calibration ROC AUC относительно seed-42 baseline хотя
  бы на `0.015`;
- object ROC AUC не хуже baseline более чем на `0.01`;
- отсутствие деградации source-image ROC AUC;
- максимум top-k не должен быть одиночным необъяснимым выбросом сетки.

Если проходят несколько вариантов, выбирается максимальный primary ROC AUC;
при разнице не более `0.01` выбирается более простой вариант: без сильной
аугментации, затем меньший train subset. Выбранный вариант повторяется на
seed 123 и 2025 и сравнивается с уже сохранёнными baseline-моделями этих
seed. Если gate не проходит никто, дополнительные обучения не запускаются.

Это exploratory selection среди пяти сравнений. Даже повторение на трёх seed
не превращает текущий раскрытый test в независимое подтверждение. Результаты
всех вариантов публикуются, а не только победителя.

## Критерии решения

Улучшение считается устойчивым, если направление эффекта сохраняется на
paired seeds и на source-group метриках, а не только на отдельных tiles.
Помимо ROC AUC обязательно сравниваются FPR/FNR при calibration threshold,
object/source errors, loss dynamics и доля независимых source groups.

Окончательный production-вывод требует нового holdout из будущих съёмок.

## Статус выполнения

Протокол выполнен 2026-07-26 без data-study test inference. Победитель
first pass `deit_data_strong_aug_1000` повторён на seed 123/2025. Большой
primary-прирост seed 42 относительно no-aug DeiT не воспроизвёлся:
post-selection mean delta `-0.000170`. При этом strong DeiT выше
ResNet18-384 на всех трёх calibration seed.

Дальнейшее обучение остановлено на общем лимите 15 запусков. Итоговый отчёт:
`fastflow_data_study_final_report_ru.md`.
