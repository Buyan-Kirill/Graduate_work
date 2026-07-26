# Code

Код и исследовательские ноутбуки проекта.

## Зависимости

Для подготовки датасета:

```bash
pip install numpy pillow tqdm
```

Для запуска ноутбуков с текущими экспериментами:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install anomalib==2.1.0 matplotlib numpy pandas pillow tqdm scikit-learn opencv-python scikit-image optuna ipywidgets
```

Если CUDA 12.8 недоступна или нужен CPU-режим, вместо первой команды установите PyTorch по инструкции с <https://pytorch.org/get-started/locally/>.

## Файлы

- `prepare_dataset.py` - подготовка датасета 3D-принтера: нарезка на тайлы, training/anomalous данные, masked anomalies.
- `calculator.ipynb` - быстрые проверки датасета и изображений.
- `FastFlow.ipynb` - ранние эксперименты FastFlow на MVTec.
- `FastFlow_printer.ipynb` - ранние эксперименты FastFlow на датасете 3D-принтера.
- `FastFlow_mvtec_compare.ipynb` - единый MVTec-протокол сравнения FastFlow ResNet18/DeiT и calibration-only top-k.
- `FastFlow_printer_resnet18.ipynb` - текущие printer-запуски ResNet18-256 и resolution-matched ResNet18-384 на трёх seed.
- `FastFlow_printer_deit.ipynb` - текущие printer-запуски DeiT Base Distilled 384 на тех же трёх seed.
- `FastFlow_printer_deit_legacy_recipe.ipynb` - исходный неудачный DeiT try_1 recipe на новом фиксированном split; первый проход только seed 42 и calibration.
- `FastFlow_printer_deit_current_recipe.ipynb` - новый DeiT recipe в отдельном каталоге для парного сравнения с legacy на том же split и seed.
- `FastFlow_printer_final_reproduction.ipynb` - воспроизведение зафиксированного итогового ResNet18/DeiT исследования.
- `fastflow_printer_pipeline.py` - общий pipeline загрузки manifest, обучения, calibration-only top-k и locked-test оценки.
- `run_fastflow_printer_experiments.py` - CLI для раздельных этапов train/calibration и locked test.
- `analyze_fastflow_printer_calibration.py` - компактный calibration-отчёт по seed, top-k, loss, clipping и исходным съёмкам.
- `report_fastflow_printer_calibration.py` - валидация первичных артефактов одного запуска и генерация статических графиков для ручного просмотра.
- `analyze_fastflow_printer_results.py` - парное статистическое сравнение DeiT с обоими ResNet18 baseline после locked test.
- `SuperSimpleNet_v3.ipynb` - SuperSimpleNet на MVTec.
- `SuperSimpleNet_printer.ipynb` - базовый SuperSimpleNet на 3D-принтере.
- `SuperSimpleNet_printer_head_finetune.ipynb` - SuperSimpleNet на 3D-принтере с fine-tuning.
- `SuperSimpleNet_mvtec_finetune.ipynb` - head fine-tuning SuperSimpleNet на MVTec.
- `SuperSimpleNet_mvtec_finetune copy.ipynb` - копия/вариант MVTec fine-tuning эксперимента.

## Запуск текущего FastFlow-сравнения

Для автономного запуска из корня проекта:

```powershell
.venv\Scripts\python.exe code\run_fastflow_printer_experiments.py --stage train-calibrate
.venv\Scripts\python.exe code\analyze_fastflow_printer_calibration.py
```

На этом этапе test не читается. После анализа calibration и фиксации
конфигураций:

```powershell
.venv\Scripts\python.exe code\run_fastflow_printer_experiments.py --stage test
.venv\Scripts\python.exe code\analyze_fastflow_printer_results.py
```

Ноутбуки предоставляют тот же процесс: `train_calibrate` обучает и калибрует
без test, `test` загружает веса и зафиксированный `calibration_selection.json`,
`load` читает сохранённые метрики. Перезапись запрещена по умолчанию.
Test-метрики нельзя использовать для повторного выбора top-k или параметров
обучения.

Все `FastFlow_printer*.ipynb` используют
`datasets/processed_printer_dataset_384` и manifest
`experiments/printer/dataset_v384_audit/printer_split_v2_final.csv`.
MVTec-ноутбуки намеренно остаются на `datasets/MVTecAD`: это отдельный
benchmark, а не printer training pipeline.

Контролируемое сравнение старого и нового DeiT recipe и правила последующих
data ablation описаны в
`experiments/printer/fastflow_deit_recipe_and_data_study_protocol_ru.md`.

Data-study запуск использует тот же CLI с явными путями:

```powershell
.venv\Scripts\python.exe code\run_fastflow_printer_experiments.py `
  --stage train-calibrate `
  --configs deit_data_mild_photo_1000 `
  --seeds 42 `
  --manifest-path experiments/printer/dataset_v384_audit/printer_split_v2_data_study.csv `
  --split-config-path configs/printer_split_v2_data_study.json `
  --output-dir experiments/printer/printer384_v2_data_study_summaries
```

Каждая config запускается отдельной командой и отдельным коммитом результатов,
чтобы GPU-обучение оставалось последовательным, а provenance — проверяемым.

Список всех автономных запусков и ссылки на их каталоги находятся в
`experiments/printer/printer384_v2_experiment_ledger.csv`. Внутри каждого
каталога сначала смотреть `run_note.md`, затем `calibration_report.md` и три
PNG-графика. Числа и SHA-256 первичных источников лежат в
`calibration_report.json`.
