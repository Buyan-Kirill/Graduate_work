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
| `deit_data_strong_aug_1000` | 42 | 0.948864 | +0.075682 | 0.985 | 0.880 | 73728 | проходит предварительно |
| `deit_data_balanced_500` | 42 | 0.930909 | +0.057727 | 0.955 | 0.880 | 147456 | проходит предварительно |
| `deit_data_all_1671` | 42 | 0.888295 | +0.015114 | 0.935 | 0.800 | 147456 | проходит на границе |
| `deit_data_object_uniform_1000` | 42 | 0.906591 | +0.033409 | 0.950 | 0.880 | 147456 | проходит предварительно |

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

## Run 2: strong legacy augmentation

Источник:
`fastflow_deit_data_strong_aug_1000_printer384_v2_data_study/try_1_seed_42`.

- Validation report status: `passed`.
- Train paths совпадают с baseline и mild augmentation.
- Best epoch: 40/40.
- Best normal-val loss: `146663.921387`.
- Training duration: `1441.940` seconds, на `153.826` seconds (`11.9%`)
  дольше mild augmentation.
- Peak allocated CUDA memory: `1277.873 MiB`.
- Gradient clipping fraction: `1.0` на каждой эпохе.
- Calibration primary ROC AUC: `0.9488636363636362`.
- Delta к baseline: `+0.0756818181818180`.
- Object ROC AUC: `0.985` (`+0.060` к baseline).
- Source-image ROC AUC: `0.88` (`+0.080` к baseline).
- Threshold balanced accuracy/FPR/FNR:
  `0.870455 / 0.05 / 0.209091`.
- Выбран top-k `73728` (`0.5` map). Соседние значения также высоки:
  `0.936023` на 49152 и `0.942500` на полном map. Максимум не выглядит
  одиночным выбросом.

Интерпретация: гипотеза о том, что сильная legacy augmentation обязательно
мешает, на seed 42 опровергнута. Ranking и threshold recall улучшились, хотя
normal-val likelihood loss выше. Это подтверждает, что абсолютные loss между
разными train distributions нельзя использовать как прямую меру anomaly
ranking. Результат пока лидирует, но один seed и пять адаптивно сравниваемых
вариантов не исключают winner's curse.

## Run 3: date-balanced 500

Источник:
`fastflow_deit_data_balanced_500_printer384_v2_data_study/try_1_seed_42`.

- Validation report status: `passed`.
- Train: 500 tiles, 167 source groups, 397 object groups.
- Best epoch: 40/40.
- Best normal-val loss: `134476.949219`.
- Training duration: `675.552` seconds; в `1.91` раза быстрее mild-1000.
- Calibration primary ROC AUC: `0.9309090909090909`.
- Delta к baseline: `+0.0577272727272727`.
- Object/source-image ROC AUC: `0.955 / 0.88`.
- Threshold balanced accuracy/FPR/FNR:
  `0.840909 / 0.05 / 0.268182`.
- Полный map снова является плавным глобальным максимумом top-k.

Интерпретация: меньший и более равномерный по датам subset лучше baseline на
seed 42, несмотря на меньшее число unique groups и вдвое меньше optimizer
steps. Это совместимо с гипотезой о вреде повторяющихся tiles, но не доказывает
её: одновременно изменились data composition, data volume и число шагов.

## Run 4: all 1671 normal tiles

Источник:
`fastflow_deit_data_all_1671_printer384_v2_data_study/try_1_seed_42`.

- Validation report status: `passed`.
- Train: 1671 tiles, 235 source groups, 800 object groups.
- Best epoch: 40/40.
- Best normal-val loss: `72784.560753`, лучший среди текущих first-pass runs.
- Training duration: `1892.428` seconds.
- Calibration primary ROC AUC: `0.8882954545454544`.
- Delta к baseline: `+0.0151136363636362`.
- Object/source-image ROC AUC: `0.935 / 0.8`.
- Threshold balanced accuracy/FPR/FNR:
  `0.677273 / 0.05 / 0.595455`.
- Full-map top-k — плавный глобальный максимум.

Интерпретация: все данные проходят primary gate только на границе и дают
намного меньший прирост, чем date-balanced 500. Более низкий normal-val loss
не перешёл в лучшее ranking/threshold качество. Наблюдение согласуется с
гипотезой о размывании normal distribution повторяющимися tiles крупнейших
дат, но также смешано с большим числом optimizer steps.

## Run 5: object-uniform 1000

Источник:
`fastflow_deit_data_object_uniform_1000_printer384_v2_data_study/try_1_seed_42`.

- Validation report status: `passed`.
- Train: 1000 tiles, все 235 source groups и 800 object groups.
- Дата `2025-05-28`: 583/1000 tiles.
- Best epoch: 40/40.
- Best normal-val loss: `96627.198242`.
- Training duration: `1198.088` seconds.
- Calibration primary ROC AUC: `0.906590909090909`.
- Delta к baseline: `+0.0334090909090908`.
- Object/source-image ROC AUC: `0.95 / 0.88`.
- Threshold balanced accuracy/FPR/FNR:
  `0.754545 / 0.05 / 0.440909`.
- Full-map top-k — плавный глобальный максимум.

Интерпретация: максимальное покрытие source/object groups при фиксированных
1000 tiles лучше baseline, но хуже date-balanced 500 на `-0.024318`.
Разнообразие объектов полезно не само по себе: сильный перекос одной даты
может перевесить выгоду дополнительного coverage.

## Решение после first pass

Все пять вариантов опубликованы. По заранее зафиксированному правилу выбран
`deit_data_strong_aug_1000`:

- максимальный primary ROC AUC `0.948864`;
- следующий вариант `balanced_500` ниже на `0.017955`, то есть правило
  упрощения при разнице не более `0.01` не применяется;
- object/source ROC AUC также максимальны: `0.985 / 0.88`;
- top-k максимум поддержан соседними значениями.

Strong augmentation повторяется только на seed 123 и 2025. Это последние два
разрешённых обучения: после них общий training count достигнет 15. Test не
запускается. Вывод будет сделан по трём paired calibration seeds с явным
учётом того, что вариант выбран как лучший из пяти на seed 42.

## Необучающий failure

`try_1_seed_42` для mild augmentation завершился до model construction:
Hugging Face metadata была недоступна через proxy. Локальный exact cache
проверен с process-local `HF_HUB_OFFLINE=1`; `try_2` выполнен успешно.
Системные proxy/VPN/Git настройки не менялись.

## Confirmatory repeat: strong augmentation, seed 123

Источник:
`fastflow_deit_data_strong_aug_1000_printer384_v2_data_study/try_2_seed_123`.

- Validation report status: `passed`.
- Calibration rows полностью совпадают с paired baseline seed 123.
- Best epoch: 40/40.
- Best normal-val loss: `147560.601074`.
- Calibration primary ROC AUC: `0.9595454545454545`.
- Paired baseline: `0.9553409090909091`.
- Paired delta: `+0.0042045454545454`.
- Object/source-image ROC AUC: `0.99 / 0.92`.
- Выбран top-k `73728` (`0.5` map). Соседний top-k `49152` даёт
  `0.958636`, full map — `0.950682`; максимум не является одиночным узким
  выбросом.
- Threshold balanced accuracy/FPR/FNR:
  `0.829545 / 0.05 / 0.290909`.
- У baseline threshold balanced accuracy выше: `0.865909`.
- Test не читался.

Интерпретация: преимущество ranking на независимом от выбора seed
воспроизвелось, но оно намного меньше наблюдавшихся на seed 42 `+0.075682`.
Это подтверждает существенную seed-вариативность и winner's curse после выбора
лучшего из пяти вариантов. Сильная аугментация пока улучшает ranking, но не
калибровку рабочего порога: эти утверждения нельзя подменять друг другом.
