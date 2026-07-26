# FastFlow DeiT: исследование аугментаций и train data

## Правила

- Выбор вариантов выполняется только по normal validation и calibration.
- Test inference в data study запрещён.
- Все five first-pass варианты seed 42 публикуются независимо от результата.
- Не более одного варианта может получить paired repeats seed 123/2025.
- Baseline: `deit_base_distilled_384`, seed 42, calibration primary ROC AUC
  `0.8731818181818182`, object ROC AUC `0.925`, source-image ROC AUC `0.8`.

## First-pass результаты

| Config | Seed | Primary ROC AUC | Delta к baseline | Object AUC | Source AUC | Top-k | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `deit_data_mild_photo_1000` | 42 | 0.890341 | +0.017159 | 0.925 | 0.800 | 147456 | проходит предварительно |

## Run 1: mild photometric augmentation

Источник:
`fastflow_deit_data_mild_photo_1000_printer384_v2_data_study/try_2_seed_42`.

- Validation report status: `passed`.
- Train: 1000 tiles, 212 source groups, 687 object groups.
- Selected train paths SHA-256:
  `3cd670c6d8cb4ed15a0ecf6cbab09ddf55dd507d423def8ac69464aef0fe3aae`.
- Best epoch: 40/40.
- Best normal-val loss: `99511.305054`.
- Training duration: `1288.114` seconds.
- Peak allocated CUDA memory: `1277.873 MiB`.
- Gradient clipping fraction: `1.0` на каждой эпохе.
- Calibration primary ROC AUC: `0.8903409090909091`.
- Object ROC AUC: `0.925`.
- Source-image ROC AUC: `0.8`.
- Threshold balanced accuracy/FPR/FNR:
  `0.763636 / 0.05 / 0.422727`.
- Full-map top-k `147456` — глобальный максимум. Primary ROC AUC возрастает
  от `0.844659` на половине map до `0.890341` на полном map; максимум не
  является одиночным узким выбросом.

Интерпретация: слабая фотометрическая аугментация проходит заранее
зафиксированный gate на seed 42, но прирост `+0.017159` мал относительно
известной seed-вариативности DeiT. Выбор победителя откладывается до завершения
всех пяти first-pass запусков.

## Необучающий failure

`try_1_seed_42` для mild augmentation завершился до model construction:
Hugging Face metadata была недоступна через proxy. Локальный exact cache
проверен с process-local `HF_HUB_OFFLINE=1`; `try_2` выполнен успешно.
Системные proxy/VPN/Git настройки не менялись.
