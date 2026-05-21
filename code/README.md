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
pip install anomalib matplotlib numpy pillow tqdm scikit-learn opencv-python scikit-image optuna ipywidgets
```

Если CUDA 12.8 недоступна или нужен CPU-режим, вместо первой команды установите PyTorch по инструкции с <https://pytorch.org/get-started/locally/>.

## Файлы

- `prepare_dataset.py` - подготовка датасета 3D-принтера: нарезка на тайлы, training/anomalous данные, masked anomalies.
- `calculator.ipynb` - быстрые проверки датасета и изображений.
- `FastFlow.ipynb` - ранние эксперименты FastFlow на MVTec.
- `FastFlow_printer.ipynb` - ранние эксперименты FastFlow на датасете 3D-принтера.
- `SuperSimpleNet_v3.ipynb` - SuperSimpleNet на MVTec.
- `SuperSimpleNet_printer.ipynb` - базовый SuperSimpleNet на 3D-принтере.
- `SuperSimpleNet_printer_head_finetune.ipynb` - SuperSimpleNet на 3D-принтере с fine-tuning.
- `SuperSimpleNet_mvtec_finetune.ipynb` - head fine-tuning SuperSimpleNet на MVTec.
- `SuperSimpleNet_mvtec_finetune copy.ipynb` - копия/вариант MVTec fine-tuning эксперимента.
