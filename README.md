# Обнаружение и локализация дефектов при лазерном сплавлении металлического порошка

Проект про anomaly detection для изображений аддитивного производства: нужно находить дефекты и локализовать их на карте аномалий. `MVTec AD` используется как стенд для проверки гипотез, основной прикладной датасет - изображения 3D-принтера.

## Навигация

- `code/` - код и исследовательские ноутбуки: подготовка данных, обучение, оценка моделей.
- `datasets/` - локальные датасеты. Полные данные лучше не хранить в GitHub.
- `experiments/` - результаты запусков: метрики, веса, графики, визуализации.
- `SSN_extention.pdf` - материал по SuperSimpleNet.

## Датасеты

- `datasets/MVTecAD/` - распакованный MVTec AD. Используется для быстрой проверки гипотез по anomaly detection и pixel-level локализации.
- `datasets/filtered_printer_dataset/` - сырые/отфильтрованные данные 3D-принтера перед нарезкой. Здесь лежат исходные изображения, маски и аномальные примеры в структуре, с которой работает `prepare_dataset.py`.
- `datasets/processed_printer_dataset/` - подготовленная выборка для обучения. Получается из `filtered_printer_dataset` после нарезки изображений на тайлы и раскладки по train/anomaly/validation-папкам.

## Сборка датасета 3D-принтера

Исходная точка - `datasets/filtered_printer_dataset/`.

Ожидаемая структура:

- `training_data/` - нормальные или размеченные тренировочные данные, обычно разложенные по датам/папкам печати.
- `anomalous_data/` - изображения с дефектами.
- маски рядом с исходными изображениями или в отдельных подпапках, если они есть.

Подготовка выполняется скриптом:

```bash
python code/prepare_dataset.py --mode all --masked_anomalies
```

Результат сохраняется в `datasets/processed_printer_dataset/`. Именно эту папку используют ноутбуки обучения на данных 3D-принтера.

## Что смотреть

- MVTec-гипотезы: `experiments/metal_nut/`, `experiments/hazelnut/`.
- Ранние backbone/FastFlow проверки: `experiments/3D_printer_wide_resnet/`, `experiments/3D_printer_resnet_18/`, `experiments/3D_printer_transformer/`.
- Текущие SuperSimpleNet эксперименты: `experiments/3D_printer_supersimplenet/`, `experiments/3D_printer_supersimplenet_head_finetune/`.

## В GitHub

В репозиторий кладётся код, README, метрики, конфиги и небольшие примеры. Полные датасеты, веса моделей, архивы и виртуальные окружения лучше исключаются через `.gitignore` и хранятся локально.
