# Цели проекта

Обновлено: 2026-07-30.

## Основная цель

Разработать и честно оценить anomaly detection/localization для поверхностных
дефектов металлических деталей, полученных на 3D-принтере.

## Текущая FastFlow-цель

Получить FastFlow с DeiT backbone качества не ниже FastFlow ResNet18 на
printer dataset либо доказательно установить, почему это не достигается.

Результат считается инженерно полезным только если:

- train/calibration/test или новый holdout разделены по исходным
  source/object groups и exact hashes;
- preprocessing, split и оценка сопоставимы между backbone;
- top-k и threshold выбираются без использования confirmatory holdout;
- показаны ranking и operating-point метрики;
- учтена вариативность по seed и зависимость tiles внутри одной съёмки;
- любой вывод воспроизводится из сохранённых кода, config, scores и metrics;
- ограничения малой выборки и domain shift явно сохранены.

## Роль MVTec AD

MVTec AD — sanity benchmark для воспроизводимости и backbone/recipe-проверок.
Он не является доказательством качества на деталях 3D-принтера.

## Приоритет метрик

- Primary ranking: source-group-balanced tile ROC AUC.
- Secondary ranking: object ROC AUC и source-image ROC AUC.
- Operating point: balanced accuracy, FPR и FNR при threshold, выбранном на
  normal calibration.
- Для uncertainty: paired cluster bootstrap по source groups и seed.
- Tile count нельзя интерпретировать как число независимых наблюдений.

## Производственные критерии

Помимо AUC учитывать:

- false negatives на целевых дефектах;
- false positives на hard negatives;
- устойчивость к новым сессиям съёмки;
- воспроизводимость;
- inference cost и GPU memory;
- прозрачность post-processing и возможность мониторинга деградации.

## Источники подробностей

- `README.md`
- `experiments/printer/fastflow_printer384_v2_protocol.md`
- `experiments/printer/fastflow_data_study_final_report_ru.md`
- `experiments/printer/fastflow_data_study_hypotheses_ru.md`
- `PROJECT_HANDOFF.md`
