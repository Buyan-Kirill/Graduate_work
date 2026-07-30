# Журнал решений и инцидентов

Формат записи:

`YYYY-MM-DD | тип | факт/решение | источник или последствие`

## 2026-07

- 2026-07-22 | решение | Зафиксирован printer split v2 с разделением по
  source/object groups и exact hashes. Top-k выбирается на calibration,
  threshold — по normal calibration. | `configs/printer_split_v2_final.json`,
  `experiments/printer/dataset_v384_audit/`.
- 2026-07-22 | domain review | Исключён один out-of-scope anomaly tile и две
  неоднозначные normal object groups; остальные отклонения приняты как hard
  negatives. | `printer_split_v2_final_domain_review.md`.
- 2026-07-23 | результат | На frozen test DeiT в среднем не показал прежнего
  сильного отставания от ResNet18, но threshold errors и uncertainty остались
  существенными. Тест после этого считается раскрытым. |
  `fastflow_printer384_v2_final_report_ru.md`.
- 2026-07-23 | отрицательный результат | Удаление gradient clipping снизило
  normal-val loss, но почти не изменило primary AUC и ухудшило threshold
  behavior. Clipping не признан причиной ranking gap. |
  `fastflow_printer384_v2_hypotheses_ru.md`.
- 2026-07-26 | решение | Data-study first pass заранее ограничен пятью
  вариантами на seed 42; повторять разрешено только одного победителя на seed
  123/2025; test inference запрещён. |
  `fastflow_deit_recipe_and_data_study_protocol_ru.md`.
- 2026-07-26 | инцидент | Первый mild-augmentation attempt завершился до
  model construction из-за недоступности Hugging Face metadata через proxy.
  Ни одного train batch не выполнено. Повтор использовал проверенный локальный
  cache и process-local `HF_HUB_OFFLINE=1`; системные настройки не менялись. |
  `fastflow_data_study_ledger.csv`.
- 2026-07-26 | результат | Strong augmentation выиграла exploratory seed 42,
  но post-selection primary delta против no-augmentation на seed 123/2025
  составила в среднем `-0.000170`. Большой seed-42 эффект признан
  winner's-curse/seed-sensitive. |
  `fastflow_data_study_final_report_ru.md`.
- 2026-07-26 | результат | Strong DeiT выше ResNet18-384 на 3/3 calibration
  seed; post-selection mean delta `+0.021023`, но cluster-bootstrap CI включает
  ноль из-за малого числа source groups. Результат признан candidate evidence,
  а не независимым подтверждением. | `paired_comparisons.csv`.
- 2026-07-26 | stop condition | Выполнено 15 FastFlow training runs суммарно.
  Новое обучение остановлено. | `printer384_v2_experiment_ledger.csv`,
  `fastflow_data_study_ledger.csv`.
- 2026-07-30 | аудит | В размеченном 384-пуле найдены только одна полностью
  неиспользованная normal source group (1 tile) и одна anomaly source group
  `L0408_1` (24 tiles). Этого недостаточно для meaningful holdout. |
  Read-only hash/source-group audit.
- 2026-07-30 | аудит | `2025-06-30_bad`: 244 уникальных tiles, 7 captures,
  63 objects, hash overlap с manifest равен нулю. Визуально пул неоднороден,
  класс не подтверждён. Нужна ручная domain review. | Read-only filesystem
  audit; данные не изменялись.
- 2026-07-30 | инцидент | Аудит raw `2025-03-21` встретил служебный файл
  `._184.png`, который PIL не распознаёт. Это metadata-файл, а не повреждение
  реального изображения. Аудит остановлен по новой инструкции пользователя. |
  Никакие dataset/split files не создавались и не изменялись.

## Правила ведения

- Добавлять только важные решения, проверенные результаты и ошибки, влияющие
  на дальнейшую работу.
- Не превращать журнал в полный terminal log.
- Для метрик указывать первичный артефакт.
- Исправления делать новой записью, не скрывая прежнюю ошибку.
