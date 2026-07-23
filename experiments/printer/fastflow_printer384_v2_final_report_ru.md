# Итоговый отчёт: FastFlow ResNet18 и DeiT на датасете принтера

## Краткий вывод

На зафиксированной тестовой выборке прежнее отставание FastFlow с DeiT не
воспроизвелось. По основной метрике ранжирования DeiT оказался выше ResNet18
на каждом из трёх заранее выбранных seed:

| Seed | ResNet18 ROC AUC | DeiT ROC AUC | DeiT - ResNet18 |
|---:|---:|---:|---:|
| 42 | 0.905938 | 0.927615 | +0.021676 |
| 123 | 0.932792 | 0.975129 | +0.042336 |
| 2025 | 0.877060 | 0.945411 | +0.068351 |

Средние результаты:

- ResNet18: `0.905263 ± 0.027872`;
- DeiT: `0.949385 ± 0.024005`;
- средняя парная разница: `+0.044121`;
- 95% paired hierarchical-bootstrap интервал:
  `[-0.034323, 0.146068]`.

Точечная оценка и все три seed говорят в пользу DeiT. Однако доверительный
интервал включает ноль. При 9 нормальных и 7 аномальных независимых source
groups нельзя утверждать, что превосходство статистически доказано.

Результат зависит от единицы принятия решения:

- по ранжированию tiles DeiT лучше;
- по зафиксированному tile threshold DeiT хуже на seed 42 и 2025;
- после max-агрегации по детали или исходной съёмке DeiT не хуже ResNet18 ни
  на одном seed и обычно делает меньше ошибок.

Таким образом, цель достигнута для ranking-качества и object/source-level
детекции, но не достигнута для заранее сформулированного per-tile threshold
критерия.

## Зафиксированный эксперимент

- Split: `printer_384_v2_final`.
- Manifest:
  `dataset_v384_audit/printer_split_v2_final.csv`.
- SHA-256 manifest:
  `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
- Train: 1000 нормальных tiles, 212 source groups.
- Normal validation: 160 tiles, 44 source groups.
- Calibration: 40 normal + 21 anomaly tiles, 5 + 5 source groups.
- Test: 53 normal + 45 anomaly tiles, 9 + 7 source groups.
- Между split нет пересечений по source group, object group и SHA-256.
- Все тестовые запуски выполнены после calibration freeze-коммита `b2893c7`.
- После просмотра test не менялись модель, top-k или threshold.

Основные конфигурации:

| Параметр | ResNet18 | DeiT distilled base |
|---|---:|---:|
| Вход | 384x384 | 384x384 |
| Epochs | 30 | 40 |
| Flow steps | 10 | 8 |
| Hidden ratio | стандартный | 0.5 |
| Learning rate | 1e-3 | 3e-5 |
| Weight decay | 1e-3 | 1e-5 |
| Batch size | 8 | 10 |
| Gradient clipping | нет | 10.0 |
| Train augmentation | нет | нет |

Обе модели создавались через Anomalib `FastflowModel(pre_trained=True)`, как
в исходных ноутбуках. Предобученный feature extractor был заморожен.

## Как проводилось сравнение

### 1. Выбор checkpoint

Модели обучались только на normal train. Лучший checkpoint выбирался по
likelihood loss на отдельной `normal_val_loss`, не содержащей source/object
groups из train. Calibration и test при выборе checkpoint не использовались.

### 2. Выбор top-k

Для каждого checkpoint на labeled calibration проверялась заранее
зафиксированная сетка top-k. Основной критерий выбора:

`source-group-balanced tile ROC AUC`.

При равенстве использовались object/tile ROC AUC, AP и затем меньший `k`.
Во всех шести основных запусках глобальным максимумом оказался полный map:
`147456 / 147456` pixels. Test для выбора top-k не использовался.

### 3. Выбор threshold

Threshold равен 95-му перцентилю anomaly score нормальных calibration tiles.
Он выбирался без test и без оптимизации по calibration anomalies.

Это правило контролирует normal FPR около 5%, но не обязано максимизировать
recall, F1 или balanced accuracy. Поэтому одновременно возможны высокий ROC
AUC и большое число FN.

### 4. Основная метрика

Основная метрика — source-group-balanced tile ROC AUC. Вклад каждого source
group нормируется, чтобы серия похожих tiles из одной съёмки не доминировала
над более малочисленными source groups.

Дополнительно считались:

- обычные tile ROC AUC и AP;
- object-max ROC AUC/AP и threshold errors;
- source-image-max ROC AUC/AP и threshold errors;
- balanced accuracy, FPR и FNR.

### 5. Парное сравнение

ResNet18 и DeiT обучались на одинаковых seed `42`, `123`, `2025`. Для каждого
seed вычислялась разница:

`DeiT primary ROC AUC - ResNet18 primary ROC AUC`.

Итоговый 95% интервал получен paired hierarchical bootstrap:

1. training seeds сэмплировались с возвращением;
2. normal и anomaly source groups сэмплировались раздельно с возвращением;
3. один и тот же bootstrap-план применялся к ResNet18 и DeiT;
4. внутри выбранных source groups сохранялись связанные tiles;
5. выполнено 2000 итераций с bootstrap seed `20260722`.

Парность уменьшает шум от конкретного состава test. Иерархическая
ресэмплизация учитывает, что tiles одной съёмки не являются независимыми.

### 6. Проверка источников

Финальный анализатор:

- повторно вычислил primary ROC AUC из каждого `test_scores.csv`;
- проверил совпадение top-k и threshold с
  `calibration_selection.json`;
- проверил одинаковый manifest, split и порядок test rows;
- сохранил отдельные tile/object/source matrices с score и prediction;
- не использовал non-inferiority margin, который не был задан заранее.

## Результаты calibration

| Seed | ResNet18 AUC | DeiT AUC | DeiT - ResNet18 |
|---:|---:|---:|---:|
| 42 | 0.919886 | 0.873182 | -0.046705 |
| 123 | 0.936818 | 0.955341 | +0.018523 |
| 2025 | 0.918750 | 0.942614 | +0.023864 |

- Среднее ResNet18: `0.925152`, sample SD `0.010120`.
- Среднее DeiT: `0.923712`, sample SD `0.044221`.
- Средняя разница: `-0.001439`.

Calibration показывает почти одинаковое среднее, но более высокую
seed-чувствительность DeiT. Выбор только seed 42 или только seed 123 привёл бы
к противоположным выводам.

## Результаты locked test

| Seed | Tile errors R/D | Object errors R/D | Source errors R/D |
|---:|---:|---:|---:|
| 42 | 14 / 28 | 7 / 5 | 4 / 4 |
| 123 | 14 / 10 | 5 / 3 | 4 / 3 |
| 2025 | 17 / 27 | 7 / 2 | 4 / 2 |

Описательные суммы по трём повторным seed-оценкам:

- tile errors: ResNet18 `45`, DeiT `65`;
- object-max errors: ResNet18 `19`, DeiT `10`;
- source-max errors: ResNet18 `12`, DeiT `9`.

Это не 3 независимых test-набора, а три модели, проверенные на одних и тех же
данных. Суммы показывают устойчивость решения к seed, а не увеличивают
эффективный размер test.

## Проверенные гипотезы

| Гипотеза | Результат | Краткий вывод |
|---|---|---|
| Переход ResNet18 с 256 на 384 заметно улучшит качество | Не подтверждена | На seed 42 primary AUC изменился только на `+0.005227`, source AUC не изменился |
| Gradient clipping мешает DeiT и объясняет отставание | Не подтверждена как причина ranking gap | Без clipping normal-val loss улучшился на `20.7%`, но primary AUC только на `+0.002159` |
| DeiT фундаментально хуже из-за представления/пространственного разрешения | Не подтверждена | На locked test DeiT выше на всех трёх seed |
| Найденный top-k является устойчивым | Подтверждена для текущего split | Полный map — argmax во всех шести основных запусках |
| Слабые tile decisions DeiT вызваны threshold, а не ranking | Подтверждена | ROC AUC выше, но q95-normal threshold даёт высокий FNR на двух seed |
| Один seed достаточен для вывода | Опровергнута | Calibration ordering меняет знак между seed 42 и 123 |

Подробный разбор доказательств:
`fastflow_printer384_v2_hypotheses_ru.md`.

## Почему старый результат DeiT был хуже

Достоверно установлено:

1. Устойчивое архитектурное отставание не воспроизводится.
2. Clipping не является причиной ranking gap.
3. Top-k не требует узкого хвоста: оптимален полный map.
4. Маленькая calibration и training seed способны менять порядок моделей.
5. Tile threshold решает другую задачу, чем ROC AUC, и имеет высокий FNR.

Точно разложить старый разрыв по причинам уже нельзя: одновременно изменились
split, контроль leakage, top-k logic, matched resolution, seed protocol и
процедура оценки. Приписать весь эффект одному изменению было бы
необоснованно.

## Ограничения

- Эффективный test содержит только 16 независимых source groups.
- Доверительный интервал разницы включает ноль.
- Нет pixel masks, поэтому качество локализации не измерено.
- Calibration содержит только 10 независимых source groups.
- Test уже раскрыт и больше не может использоваться для честного tuning.
- Не проверена генерализация на будущие производственные сессии.

## Проверка итогового notebook

Notebook:
`../../code/FastFlow_printer_final_reproduction.ipynb`.

Проведена статическая проверка без выполнения его ячеек:

- notebook является корректным JSON `nbformat 4.5`;
- все 8 code cells проходят `ast.parse`;
- сохранённых outputs нет;
- внутри notebook нет собственной реализации optimizer, loss или training
  loop;
- обучение, calibration и test вызывают тот же
  `fastflow_printer_pipeline.run_experiment`;
- frozen results загружаются через тот же `load_experiment_result`;
- итоговое сравнение пересчитывается через тот же
  `analyze_fastflow_printer_results.compare_runs`;
- параметры сравнения совпадают с финальным запуском:
  `2000` iterations, bootstrap seed `20260722`,
  `max_extra_tile_errors=1`, non-inferiority margin не задан;
- список `FROZEN_RUNS` точно соответствует шести фактическим run directories;
- для всех шести runs совпадают config, seed, try, manifest, top-k и threshold;
- текущие `PRINTER_CONFIGS` полностью совпадают с сохранёнными run configs.

Provenance-нюанс:

- seed 42 был обучен с pipeline SHA-256 `358054...`;
- seeds 123/2025 — с `47ac8a...`, который используется сейчас;
- Git diff между версиями содержит только добавление отдельного
  `deit_base_distilled_384_no_clip` config и его описания: 9 новых строк;
- training, inference, primary ResNet/DeiT configs и metric functions не
  менялись.

Следовательно, текущий notebook воспроизводит фактический финальный pipeline,
а не его упрощённую копию. Полный повтор шести обучений намеренно выполняется
последовательно: один config/seed, commit артефактов, затем следующий run.
Это сохраняет provenance и не позволяет запускать GPU training параллельно.

Для численного повторения необходимы тот же manifest, локальные cached
pretrained weights, версии библиотек и сопоставимое CUDA/GPU окружение.
Детерминированный режим включён, но bitwise-совпадение на другом hardware или
других версиях CUDA/PyTorch не гарантируется.

## Следующие шаги

1. Не менять threshold по текущему test.
2. Добавить независимые normal/anomaly source captures в calibration.
3. Заранее определить целевую функцию threshold:
   ограничение на FPR, минимальный recall или стоимость FP/FN.
4. Проверить новый threshold один раз на новом future-session holdout.
5. Определить production unit: отдельный tile, деталь или исходная съёмка.
6. При необходимости разметить небольшой набор pixel masks для проверки
   локализации.

## Артефакты

- Полный реестр:
  `fastflow_printer384_v2_hypotheses_ru.md`.
- Машиночитаемый paired report:
  `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison.json`.
- Метрики по seed:
  `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison_per_seed.csv`.
- Score matrices:
  `printer384_v2_final_test_comparison_v2/*_tile_scores.csv`,
  `*_object_scores.csv`, `*_source_scores.csv`.
- Notebook:
  `../../code/FastFlow_printer_final_reproduction.ipynb`.
