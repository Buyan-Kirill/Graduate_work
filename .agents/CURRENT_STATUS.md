# Актуальное состояние

Обновлено: 2026-07-30.

## FastFlow printer

- Approved dataset root:
  `datasets/processed_printer_dataset_384`.
- Data-study manifest:
  `experiments/printer/dataset_v384_audit/printer_split_v2_data_study.csv`.
- Manifest SHA-256:
  `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
- Calibration: 61 tiles, 10 source groups.
- Test: 98 tiles. Он уже раскрыт в предыдущем backbone study и не является
  новым независимым holdout.
- Data study завершён без test inference. Выполнено 7 успешных data-study
  обучений; общий счётчик FastFlow study достиг hard stop 15.

## Последний подтверждённый результат

По трём calibration seed:

| Config | Primary mean | Sample SD |
|---|---:|---:|
| DeiT strong augmentation | 0.948826 | 0.010739 |
| ResNet18-384 | 0.925152 | 0.010120 |
| DeiT no augmentation | 0.923712 | 0.044221 |

Strong DeiT выше ResNet18 на 3/3 calibration seed. Однако относительно
no-augmentation DeiT post-selection mean delta на seed 123/2025 равна
`-0.000170`. Большой прирост seed 42 не воспроизвёлся как устойчивый эффект
аугментации.

Источники:

- `experiments/printer/fastflow_data_study_final_report_ru.md`
- `experiments/printer/printer384_v2_data_study_analysis/paired_runs.csv`
- `experiments/printer/printer384_v2_data_study_analysis/paired_comparisons.csv`

## Что означает strong

`strong` — название bundled train augmentation
`legacy_deit_try1`, а не более сильная модель. Bundle включает crop, flips,
rotation, ColorJitter, частый blur и RandomErasing. Отдельные компоненты не
аблировались.

## Data-volume ablation

500/1000/1671 train variants обучались по 40 epochs с batch size 10.
Optimizer steps не выравнивались:

- 500 tiles: около 2000 steps;
- 1000 tiles: около 4000 steps;
- 1671 tiles: около 6720 steps.

Поэтому текущий результат сравнивает полные recipes при fixed epochs, но не
изолирует чистый эффект объёма или однородности данных.

## Незавершённый read-only аудит нового holdout

На 2026-07-30 никаких новых split/review folders не создано.

Подтверждено:

- в основном 384-пуле полностью не использована только одна normal source
  group `normal:2025-05-28:181` с одним tile;
- полностью не использована одна anomaly source group
  `anomaly:2025-08-06_11-28-56_L0408_1` с 24 tiles;
- остальные 958 формально unused anomaly tiles в основном относятся к уже
  использованным source groups и не являются независимым holdout;
- все 14 heldout-normal captures из дат `2025-02-07_test` и
  `2025-06-25_test` уже представлены в calibration/test;
- `2025-06-30_bad` содержит 244 уникальных tiles, 7 новых captures и 63
  objects без hash overlap с manifest, но визуально неоднороден и не имеет
  подтверждённой бинарной разметки;
- raw-сессия `2025-03-21` требует отдельного аудита: processed folders пусты,
  а read-only проверка raw files была прервана на служебном macOS-файле
  `._184.png`. Реальные изображения не изменялись.

Текущий вывод: из уже размеченного processed pool нельзя составить
статистически содержательный новый holdout без привлечения/разметки
`2025-06-30_bad`, восстановления `2025-03-21` либо новых съёмок.

## Git

- Agent context введён опубликованным commit:
  `22590ce Add agent project context and operating rules`.
- Последний опубликованный data-study commit:
  `497612f Finalize FastFlow DeiT data study`.
- После отправки `22590ce` в `main` рабочее дерево было чистым.
- Для push использовать прямой HTTPS URL из `AGENTS.md`; сохранённый `origin`
  может указывать на SSH. Git/proxy/VPN settings не менять.
