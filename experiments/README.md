# Experiments

Результаты запусков по anomaly detection и локализации дефектов.

## Папки

- `metal_nut/` - MVTec `metal_nut`, проверка SuperSimpleNet и настроек обучения.
- `hazelnut/` - MVTec `hazelnut`, проверка SuperSimpleNet и head fine-tuning.
- `3D_printer_wide_resnet/` - ранние FastFlow/backbone-запуски с Wide ResNet.
- `3D_printer_resnet_18/` - ранние FastFlow/backbone-запуски с ResNet-18.
- `3D_printer_transformer/` - ранняя проверка transformer-подхода.
- `printer/` - исторические и текущие FastFlow-запуски на датасете принтера; новый protocol использует отдельные теги `*_printer384_v2_final` и не перезаписывает legacy-результаты.
- `3D_printer_supersimplenet/` - базовый SuperSimpleNet на датасете 3D-принтера.
- `3D_printer_supersimplenet_head_finetune/` - текущие эксперименты SuperSimpleNet с fine-tuning.

## Типовые файлы

- `final_metrics.txt` - итоговые метрики запуска.
- `middle_logs/` - чекпоинты, промежуточные метрики, визуализации по эпохам.
- `trained_model*.pth` - сохраненные модели/веса.
- `*.png` - графики loss/metrics и примеры карт аномалий.

## Current FastFlow printer protocol

- Полное описание: `printer/fastflow_printer384_v2_protocol.md`.
- Итоговый отчёт ResNet18/DeiT: `printer/fastflow_printer384_v2_final_report.md`.
- Зафиксированный split и аудит: `printer/dataset_v384_audit/`.
- Top-k выбирается только на labeled calibration; threshold — только по normal calibration scores.
- Test оценивается один раз с зафиксированным post-processing.
- Основная метрика сравнения — source-image-balanced tile ROC AUC; дополнительно сохраняются обычные tile, object и source-image метрики.
- Неопределённость оценивается cluster bootstrap по исходным съёмкам и парным bootstrap по трём training seed.
- Реестр автономных запусков: `printer/printer384_v2_experiment_ledger.csv`.
- Для ручной проверки каждый новый run содержит описание, source-validated calibration report и статические графики loss, top-k и scores.


## Current SSN evaluation logic

- MVTec SSN runs use a clean normal `train/val` split for synthetic `Validation_Loss`.
- Labeled segmentation/classification data, when available, is split separately into metric `val/test` sets.
- Missing labeled tasks can fall back to synthetic metrics from held-out normal images; these must be named `synthetic_*` and treated as diagnostics, not real benchmark metrics.
- Final reports may include per-defect metrics and one-per-class visualizations; epoch logs stay compact with only aggregate metrics.
