# Обнаружение и локализация дефектов при лазерном сплавлении металлического порошка

Проект про anomaly detection для изображений аддитивного производства: нужно находить дефекты и локализовать их на карте аномалий. `MVTec AD` используется как стенд для проверки гипотез, основной прикладной датасет - изображения 3D-принтера.

## Навигация

- `code/` - код и исследовательские ноутбуки: подготовка данных, обучение, оценка моделей.
- `datasets/` - локальные датасеты; структура описана в [datasets/README.md](datasets/README.md). Полные данные лучше не хранить в GitHub.
- `experiments/` - результаты запусков: метрики, веса, графики, визуализации.
- `SSN_extention.pdf` - материал по SuperSimpleNet.

## Датасеты

Подробное описание локальных датасетов, их структуры, масок и отличий MVTec AD от датасета 3D-принтера лежит в [datasets/README.md](datasets/README.md).

Коротко:

- `datasets/MVTecAD/` - стенд для проверки гипотез и pixel-level оценки.
- `datasets/3d_printer_dataset/` - основной прикладной датасет с изображениями 3D-принтера.
- `datasets/processed_printer_dataset/` - локально подготовленная выборка для обучения.

Базовая подготовка датасета 3D-принтера:

```bash
python code/prepare_dataset.py --mode all
```

## Что смотреть

- MVTec-гипотезы: `experiments/metal_nut/`, `experiments/hazelnut/`.
- Ранние backbone/FastFlow проверки: `experiments/3D_printer_wide_resnet/`, `experiments/3D_printer_resnet_18/`, `experiments/3D_printer_transformer/`.
- Текущие SuperSimpleNet эксперименты: `experiments/3D_printer_supersimplenet/`, `experiments/3D_printer_supersimplenet_head_finetune/`.

## В GitHub

В репозиторий кладётся код, README, метрики, конфиги и небольшие примеры. Полные датасеты, веса моделей, архивы и виртуальные окружения лучше исключаются через `.gitignore` и хранятся локально. Файл `datasets/README.md` специально разрешён в `.gitignore`, чтобы описание датасетов попадало в GitHub без самих данных.
