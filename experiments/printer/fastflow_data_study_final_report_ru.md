# FastFlow DeiT: итог исследования аугментаций и train data

Дата фиксации: 2026-07-26.

## Короткий вывод

Сильная legacy-аугментация дала наиболее устойчивый DeiT относительно
ResNet18 на текущей calibration-выборке, но не дала воспроизводимого прироста
primary ROC AUC относительно DeiT без аугментации.

- Strong DeiT: primary ROC AUC `0.948826 +/- 0.010739` по трём seed.
- ResNet18-384: `0.925152 +/- 0.010120`.
- DeiT без аугментации: `0.923712 +/- 0.044221`.
- Strong DeiT выше ResNet18 на всех трёх seed.
- На двух post-selection seed strong-minus-no-aug равен в среднем
  `-0.000170`: большой выигрыш seed 42 не воспроизвёлся.

Поэтому strong DeiT можно заморозить как кандидат для нового holdout, но
нельзя утверждать, что сильная аугментация уже доказанно улучшает DeiT.

## Валидность сравнения

- Все варианты используют одни и те же 61 calibration tile из 10 source
  groups.
- Старый baseline train и новый date-balanced subset содержат одни и те же
  1000 путей. SHA-256 отсортированного списка:
  `3cd670c6d8cb4ed15a0ecf6cbab09ddf55dd507d423def8ac69464aef0fe3aae`.
- Основная метрика: source-group-balanced tile ROC AUC.
- Top-k и q95-normal threshold выбирались только на calibration.
- Seed 42 использовался для выбора одного победителя из пяти вариантов.
- Seed 123 и 2025 были зарезервированы для post-selection replication.
- Test inference в исследовании не запускался.
- Выполнено 7 успешных data-study обучений. Один технический запуск завершился
  до model construction и не выполнил ни одного train batch.
- После последнего запуска общий лимит 15 обучений достигнут.

## Первый проход

Все строки ниже относятся к seed 42 и поэтому являются exploratory.

| Вариант | Train tiles | Primary AUC | Delta | Object AUC | Source AUC | Bal. accuracy | Top-k |
|---|---:|---:|---:|---:|---:|---:|---:|
| no augmentation | 1000 | 0.873182 | 0 | 0.925 | 0.800 | 0.711364 | 147456 |
| mild photometric | 1000 | 0.890341 | +0.017159 | 0.925 | 0.800 | 0.763636 | 147456 |
| strong legacy augmentation | 1000 | 0.948864 | +0.075682 | 0.985 | 0.880 | 0.870455 | 73728 |
| date-balanced | 500 | 0.930909 | +0.057727 | 0.955 | 0.880 | 0.840909 | 147456 |
| all available | 1671 | 0.888295 | +0.015114 | 0.935 | 0.800 | 0.677273 | 147456 |
| object-uniform | 1000 | 0.906591 | +0.033409 | 0.950 | 0.880 | 0.754545 | 147456 |

Strong augmentation прошла заранее заданный gate и была единственным вариантом,
повторённым на seed 123/2025.

## Парные повторы

| Seed | ResNet18 | DeiT no-aug | DeiT strong | Strong - no-aug | Strong - ResNet |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.919886 | 0.873182 | 0.948864 | +0.075682 | +0.028977 |
| 123 | 0.936818 | 0.955341 | 0.959545 | +0.004205 | +0.022727 |
| 2025 | 0.918750 | 0.942614 | 0.938068 | -0.004545 | +0.019318 |

Агрегированные calibration-метрики:

| Config | Primary mean | Sample SD | Object mean | Source mean | Bal. accuracy mean |
|---|---:|---:|---:|---:|---:|
| strong DeiT | 0.948826 | 0.010739 | 0.988333 | 0.906667 | 0.846212 |
| ResNet18-384 | 0.925152 | 0.010120 | 0.965000 | 0.906667 | 0.740909 |
| no-aug DeiT | 0.923712 | 0.044221 | 0.956667 | 0.840000 | 0.774242 |

Средний FNR strong DeiT равен `0.257576`, no-aug DeiT — `0.401515`,
ResNet18 — `0.468182`; FPR у всех этих calibration operating points равен
`0.05` по конструкции q95-normal threshold.

## Проверка winner's curse и top-k

Главное сравнение аугментации нужно делать по seed 123/2025, которые не
участвовали в выборе победителя:

| Scores | Strong - no-aug mean | Seed wins | Cluster-bootstrap 95% interval |
|---|---:|---:|---:|
| per-run selected top-k | -0.000170 | 1/2 | [-0.050000, 0.067500] |
| fixed full map | -0.004602 | 0/2 | [-0.056250, 0.070000] |

Следовательно, прирост `+0.075682` на seed 42 был в значительной степени
selection/seed effect. Top-k strong DeiT также не полностью стабилен:
`50%`, `50%`, `100%` map по seed 42/123/2025. При фиксированном full map
strong всё ещё выше ResNet18 на всех трёх seed, но не выше no-aug DeiT на
post-selection seed.

Для strong-minus-ResNet на seed 123/2025:

- selected top-k mean delta: `+0.021023`, два выигрыша из двух;
- fixed full-map mean delta: `+0.016591`, два выигрыша из двух;
- bootstrap interval selected top-k: `[-0.082753, 0.188892]`.

Широкий интервал не опровергает наблюдаемый эффект, но показывает, что 10
source groups недостаточно для точного статистического вывода. Bootstrap
также не исправляет bias от выбора рецепта и top-k на той же calibration.

## Что установлено о данных

1. Больше tiles не означает автоматически лучше. Вариант 1671 получил самый
   низкий normal-val loss `72784.56`, но primary AUC только `0.888295`.
2. Date-balanced 500 на seed 42 лучше 1000 и 1671 по primary AUC, но это один
   seed; одновременно изменились composition, объём данных и число optimizer
   steps. Причинный вывод о вреде дублей пока невозможен.
3. Object-uniform 1000 покрывает все 235 source groups и 800 objects, но
   уступает date-balanced 500 на `0.024318`. Coverage без контроля
   доминирующей даты недостаточно.
4. Mild photometric augmentation дала небольшой прирост primary `+0.017159`,
   но не изменила object/source AUC на seed 42.
5. Strong legacy bundle улучшил object/source и снизил наблюдаемую
   seed-вариативность, но его отдельные операции не аблировались. Нельзя
   утверждать, что RandomErasing, blur, rotation или crop полезны по
   отдельности.
6. Normal-val likelihood loss нельзя использовать как surrogate anomaly AUC
   между разными train distributions.

## Решение и следующие шаги

1. Не запускать дополнительные обучения в текущем study: лимит достигнут.
2. Заморозить strong DeiT и no-aug DeiT, их top-k/threshold и ResNet18 как
   три заранее определённых системы.
3. Собрать новый holdout из будущих независимых сессий. Не переносить туда
   source/object groups из train/calibration и не менять рецепты после
   просмотра изображений.
4. Выполнить на новом holdout только inference существующих checkpoints.
   Primary unit — source group; дополнительно показать object/source AUC,
   FPR/FNR и paired cluster bootstrap.
5. Если нового holdout пока нет, запуск на текущем раскрытом test считать
   только ретроспективной диагностикой, не подтверждением.
6. Возвращаться к component-wise augmentation и equal-step data-volume
   ablation только после появления новой calibration/holdout структуры.

## Артефакты

- [Dashboard](printer384_v2_data_study_analysis/data_study_dashboard.png)
- [Первый проход](printer384_v2_data_study_analysis/first_pass_seed42.csv)
- [Парные запуски](printer384_v2_data_study_analysis/paired_runs.csv)
- [Bootstrap-сравнения](printer384_v2_data_study_analysis/paired_comparisons.csv)
- [Machine-readable summary](printer384_v2_data_study_analysis/data_study_summary.json)
- [Журнал всех запусков](fastflow_data_study_ledger.csv)
- [Подробные промежуточные выводы](fastflow_data_study_findings_ru.md)
- [Воспроизводимый анализ](../../code/analyze_fastflow_printer_data_study.py)
- [Notebook для ручной проверки](../../code/FastFlow_printer_deit_data_study_reproduction.ipynb)
