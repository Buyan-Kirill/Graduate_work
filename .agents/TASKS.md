# Текущие задачи

Обновлено: 2026-07-30.

## P0 — ожидает решения пользователя

### Новый независимый holdout

Цель: confirmatory-сравнение frozen checkpoints:

1. ResNet18-384;
2. DeiT без аугментации;
3. DeiT strong augmentation.

Primary hypothesis:

`mean primary ROC AUC(DeiT strong) - mean primary ROC AUC(ResNet18) > 0`
на новых source groups при frozen per-seed top-k и threshold.

Secondary hypothesis:

strong augmentation снижает seed variance и улучшает object/source ranking
относительно no-augmentation DeiT.

До inference зафиксировать:

- manifest и SHA;
- primary/secondary metrics;
- source-group bootstrap;
- отсутствие recalibration/top-k selection на holdout;
- правило интерпретации широкого CI;
- все три model families и три seed.

### Доступные варианты данных

1. Предпочтительно — новые независимые сессии good/anomaly.
2. Возможный локальный резерв — вручную разметить
   `2025-06-30_bad` и проверить/обработать `2025-03-21`.
3. Единственная полностью неиспользованная размеченная anomaly-сессия
   `L0408_1` и один normal tile недостаточны для confirmatory holdout.

Для продолжения требуется разрешение пользователя на конкретные действия:

- закончить read-only аудит `2025-03-21`;
- создать review copies для `2025-06-30_bad`, `2025-03-21` и `L0408_1`;
- после ручной оценки создать proposed holdout manifest.

## P1 — возможный будущий эксперимент

### Fixed-step data-volume control

Текущий 500/1000/1671 ablation смешивает объём, composition и число updates.
Корректный контроль:

- nested date-balanced subsets;
- одинаковое число optimizer steps;
- одинаковый LR schedule в единицах steps;
- минимум два заранее выбранных seed;
- test не читать.

Этот эксперимент сейчас не разрешён: hard stop 15 обучений достигнут.

## Закрыто

- MVTec backbone sanity comparison.
- Printer split v2 audit и domain review.
- Три paired seed для ResNet18-384 и DeiT no-augmentation.
- Data-study first pass из пяти вариантов.
- Strong-augmentation repeats seed 123/2025.
- Calibration-only paired analysis, bootstrap, dashboard, русский отчёт и
  reproduction notebook.

## Запрещено без нового разрешения

- новые GPU training runs;
- перенос/удаление/перезапись данных;
- изменение существующих split;
- test-driven tuning;
- commit/push текущих agent files;
- системные/Git/proxy/VPN configuration changes.
