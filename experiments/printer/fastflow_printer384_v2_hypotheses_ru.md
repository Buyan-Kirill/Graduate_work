# Полный реестр гипотез FastFlow printer384 v2

## Назначение

Этот файл сводит гипотезы, контролируемые изменения, фактические результаты и
принятые решения. Числа берутся только из сохранённых CSV/JSON и отчётов
запусков.

Правила эксперимента:

- checkpoint выбирается по отдельной normal validation;
- top-k выбирается только на labeled calibration;
- threshold выбирается только по normal calibration scores;
- test запускается после freeze и не используется для tuning;
- ResNet18 и DeiT сравниваются попарно на seed `42`, `123`, `2025`;
- основной показатель — source-group-balanced tile ROC AUC.

## Сводная таблица гипотез

| ID | Гипотеза | Статус | Ключевое доказательство | Решение |
|---|---|---|---|---|
| H1 | Разрешение 384 само по себе заметно улучшит ResNet18 | Не подтверждена | ResNet18-384 против ResNet18-256 на seed 42: primary `+0.005227`, object `+0.010`, source `0.0` | Использовать 384 как честный matched control, не как доказанное улучшение |
| H2 | Gradient clipping 10.0 вызывает плохое обучение DeiT | Не подтверждена как причина ranking gap | No-clip улучшил normal-val loss на `20.7%`, но primary AUC только на `+0.002159`; Spearman scores `0.988102` | Не продвигать no-clip и не повторять на других seed |
| H3 | DeiT стабильно хуже из-за архитектурного представления | Не подтверждена | Locked test delta по seed: `+0.021676`, `+0.042336`, `+0.068351` | Не менять backbone/optimizer; признать ranking DeiT не хуже |
| H4 | Оптимальный top-k устойчив между seed/backbones | Подтверждена для текущего split | Полный map `147456` — primary argmax во всех 6 основных запусках | Зафиксировать full-map aggregation |
| H5 | Проблема DeiT находится в threshold calibration, а не ranking | Подтверждена, но не решена | Test AUC выше на всех seed, tile errors `28/10/27` против `14/14/17`; object/source errors не хуже | Не менять threshold на раскрытом test; собрать новый calibration/holdout |
| H6 | Одного seed достаточно для вывода | Опровергнута | На calibration seed 42 даёт delta `-0.046705`, seed 123 `+0.018523` | Использовать все 3 заранее обоснованных paired seed |

## H1. Влияние разрешения ResNet18

Проверка:

- `resnet18_256`, seed 42;
- `resnet18_384`, seed 42;
- одинаковый leak-free manifest, но соответствующий backbone input.

Факты:

- primary calibration AUC:
  `0.914659090909091 -> 0.9198863636363637`;
- изменение: `+0.005227272727272636`;
- object AUC:
  `0.955 -> 0.965`;
- source-image AUC:
  `0.88 -> 0.88`;
- оба запуска выбрали полный map.

Вывод:

384 необходимо для matched comparison с DeiT, но на этом seed нет
доказательства существенного resolution-only улучшения ResNet18.

Источники:

- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/`;
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/`.

## H2. Gradient clipping в DeiT

Причина проверки:

В baseline DeiT seed 42 clipping с нормой `10.0` срабатывал в `100%` batches.
Это означало, что clipping являлся частью optimizer dynamics, а не редкой
защитой от выбросов.

Контролируемое изменение:

- baseline: `grad_clip_norm=10.0`;
- ablation: `grad_clip_norm=None`;
- остальные параметры и seed одинаковы.

Факты:

- normal-val loss:
  `97379.98022460938 -> 77184.96472167969`;
- относительное снижение: `20.7%`;
- primary calibration AUC:
  `0.8731818181818182 -> 0.875340909090909`;
- изменение primary: `+0.0021590909090908`;
- source-image AUC: `0.80 -> 0.80`;
- threshold balanced accuracy:
  `0.71136 -> 0.70227`;
- Pearson/Spearman scores:
  `0.990304 / 0.988102`.

Вывод:

Clipping заметно меняет likelihood optimization и scale scores, но почти не
меняет порядок изображений. Он не объясняет ranking gap и no-clip не имеет
достаточного основания для продвижения.

Источники:

- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/`;
- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/`.

## H3. Архитектурный deficit DeiT

Архитектурный факт:

- ResNet18 FastFlow использует три feature scales;
- текущий DeiT FastFlow использует один patch-token feature map 24x24 при
  входе 384.

Это делало гипотезу о проблеме небольших локальных дефектов правдоподобной,
но не являлось доказательством.

Calibration:

| Seed | ResNet18 | DeiT | Delta |
|---:|---:|---:|---:|
| 42 | 0.919886 | 0.873182 | -0.046705 |
| 123 | 0.936818 | 0.955341 | +0.018523 |
| 2025 | 0.918750 | 0.942614 | +0.023864 |

Test:

| Seed | ResNet18 | DeiT | Delta |
|---:|---:|---:|---:|
| 42 | 0.905938 | 0.927615 | +0.021676 |
| 123 | 0.932792 | 0.975129 | +0.042336 |
| 2025 | 0.877060 | 0.945411 | +0.068351 |

Итог:

- test mean delta: `+0.044121`;
- paired 95% bootstrap CI:
  `[-0.034323, 0.146068]`;
- DeiT выше на всех seed;
- CI включает ноль из-за малого числа source groups.

Вывод:

Устойчивый representation deficit не подтверждён. Нельзя утверждать
статистически доказанное превосходство DeiT, но текущие данные также не дают
оснований считать DeiT хуже по ranking.

## H4. Устойчивость top-k

Проверка проведена для:

- ResNet18-384 seeds 42, 123, 2025;
- DeiT-384 seeds 42, 123, 2025;
- дополнительного no-clip diagnostic.

Факт:

Во всех шести основных запусках full map `147456 / 147456` является
глобальным primary-AUC argmax. No-clip diagnostic также выбирает full map.

Некоторые sweep curves имеют локальные немонотонные шаги. Поэтому
утверждается только endpoint optimality, а не строгая монотонность.

Вывод:

Для текущего split узкий top-k не подтверждается. Используется среднее по всей
anomaly map. Test не использовался для выбора.

Источники:

- `printer384_v2_final_calibration_comparison/*top_k_sweeps*.csv`;
- `calibration_top_k_sweep.png` в каждом run directory.

## H5. Threshold calibration

Threshold:

95-й перцентиль normal calibration tile scores.

Calibration source-balanced FNR:

| Seed | ResNet18 FNR | DeiT FNR |
|---:|---:|---:|
| 42 | 0.4182 | 0.5273 |
| 123 | 0.4864 | 0.2182 |
| 2025 | 0.5000 | 0.4591 |

Test source-balanced FNR:

| Seed | ResNet18 FNR | DeiT FNR |
|---:|---:|---:|
| 42 | 0.4102 | 0.6429 |
| 123 | 0.4782 | 0.1756 |
| 2025 | 0.5197 | 0.5775 |

Threshold уже на calibration имеет высокий FNR. Следовательно, проблема не
сводится к неожиданному calibration-test shift: q95-normal правило изначально
контролирует FPR, а не recall.

Test errors:

| Seed | Tile R/D | Object R/D | Source R/D |
|---:|---:|---:|---:|
| 42 | 14 / 28 | 7 / 5 | 4 / 4 |
| 123 | 14 / 10 | 5 / 3 | 4 / 3 |
| 2025 | 17 / 27 | 7 / 2 | 4 / 2 |

Вывод:

По tiles DeiT threshold нестабилен. При max-агрегации обычно остаётся хотя бы
один сильный anomalous tile, поэтому object/source result значительно лучше.
Новый threshold нельзя выбирать по раскрытому test.

## H6. Seed-чувствительность

Calibration sample SD:

- ResNet18: `0.010120`;
- DeiT: `0.044221`.

Test sample SD:

- ResNet18: `0.027872`;
- DeiT: `0.024005`.

На calibration DeiT значительно чувствительнее к seed, и ordering меняет
знак. На test вариативность DeiT не выше ResNet18, а paired delta положителен
во всех трёх случаях.

Вывод:

Один запуск был недостаточен для вывода. Третий paired seed был обоснован
противоречием первых двух, а не добавлен после просмотра test.

## Реестр запусков

### ResNet18-256, seed 42

- Статус: train + calibration.
- Best epoch: `14 / 14`.
- Primary calibration AUC: `0.914659090909091`.
- Top-k: full map.
- Роль: исторический reference.

### ResNet18-384, seed 42, неудачная попытка

- Статус: ошибка до model construction, training batches `0`.
- Причина: попытка timm получить Hugging Face metadata через недоступный
  proxy до использования локального cache.
- Исправление: process-local `HF_HUB_OFFLINE=1`.
- Попытка сохранена и не считается обучением.

### ResNet18-384, seeds 42/123/2025

| Seed | Best epoch | Calibration AUC | Test AUC | Training seconds |
|---:|---:|---:|---:|---:|
| 42 | 30 | 0.919886 | 0.905938 | 626.879 |
| 123 | 30 | 0.936818 | 0.932792 | 625.679 |
| 2025 | 30 | 0.918750 | 0.877060 | 625.353 |

### DeiT-384 baseline, seeds 42/123/2025

| Seed | Best epoch | Calibration AUC | Test AUC | Training seconds |
|---:|---:|---:|---:|---:|
| 42 | 40 | 0.873182 | 0.927615 | 1207.346 |
| 123 | 40 | 0.955341 | 0.975129 | 1205.167 |
| 2025 | 40 | 0.942614 | 0.945411 | 1204.815 |

Во всех baseline DeiT запусках clipping срабатывал в `100%` batches, но
обучение было численно стабильным.

### DeiT no-clip, seed 42

- Best epoch: `40 / 40`.
- Primary calibration AUC: `0.875340909090909`.
- Test не запускался, так как diagnostic не был продвинут.
- Роль: однофакторная проверка H2.

Всего:

- 8 завершённых обучений;
- 1 сохранённая ошибка до обучения;
- 6 frozen locked-test evaluations;
- не было параллельных GPU trainings;
- test не использовался для последующего tuning.

## Контроль валидности

- Manifest SHA-256 проверялся перед каждым запуском.
- Test rows совпадают между всеми шестью моделями.
- Primary metrics пересчитаны из `test_scores.csv`.
- Test top-k/threshold совпадают с `calibration_selection.json`.
- Bootstrap детерминирован seed `20260722`.
- Два запуска финального анализатора дали одинаковый JSON report.
- Веса и checkpoints хранятся локально и не добавлены в Git.
- Метрики, configs, provenance, plots и отчёты сохранены в Git.
- Итоговый notebook статически проверен: JSON/AST корректны, outputs очищены,
  configs и frozen run mapping совпадают с сохранёнными runs.
- Различие pipeline hash у seed 42 проверено через Git diff: изменилось только
  добавление no-clip diagnostic config, основной training/evaluation код не
  менялся.

## Принятое решение

1. Не запускать дополнительные training variants на текущем holdout.
2. Считать ranking-цель DeiT достигнутой, но не заявлять статистически
   доказанное превосходство.
3. Явно отметить провал per-tile threshold критерия.
4. Для production ориентироваться на заранее выбранную единицу решения:
   tile, object или source capture.
5. Любое изменение threshold подтверждать только на новом holdout.

## Основные источники

- `fastflow_printer384_v2_final_report_ru.md`;
- `fastflow_printer384_v2_findings.md`;
- `printer384_v2_experiment_ledger.csv`;
- `printer384_v2_final_calibration_comparison/`;
- `printer384_v2_final_test_comparison_v2/`;
- индивидуальные `run_note.md`, `train_history.csv`,
  `calibration_metrics.json`, `test_metrics.json` и `test_scores.csv`.
