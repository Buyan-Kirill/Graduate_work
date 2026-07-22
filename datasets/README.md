# Датасеты

Эта папка хранит локальные данные для экспериментов по anomaly detection и локализации дефектов. Полные датасеты, архивы и подготовленные выборки не добавляются в GitHub; в репозиторий должен попадать только этот README.

## Зависимости для подготовки датасета

Для запуска `code/prepare_dataset.py` нужны `numpy`, `Pillow` и опционально `tqdm` для progress bar:

```bash
pip install numpy pillow tqdm
```

## MVTec AD

Путь по умолчанию: `datasets/MVTecAD/`.

`MVTec AD` используется как стенд для быстрой проверки гипотез, отладки пайплайнов обучения и оценки pixel-level локализации. Типичная структура категории:

```text
MVTecAD/
  metal_nut/
    train/
      good/
    test/
      good/
      <defect_type>/
    ground_truth/
      <defect_type>/
  hazelnut/
    ...
```

Семантика данных:

- `train/good/` - нормальные изображения для обучения.
- `test/good/` - нормальные изображения для теста.
- `test/<defect_type>/` - аномальные изображения для теста.
- `ground_truth/<defect_type>/` - маски дефектов для аномальных тестовых изображений.

Важно: маски в `ground_truth/` являются именно масками аномалий. Их можно использовать для pixel-level метрик и визуальной проверки карты аномалий.

## Датасет 3D-принтера

Путь по умолчанию: `datasets/3d_printer_dataset/`.

Это основной прикладной датасет проекта. Он описывает изображения 3D-принтера и используется для подготовки выборки через `code/prepare_dataset.py`.

Ожидаемая структура:

```text
3d_printer_dataset/
  training_data/
    <date>/
      rect/
        <frame_id>.png
      script_masks/
        <frame_id>.png
      correlation_script_masks/
        <frame_id>.png
  anomalous_data/
    <image>.png
```

Семантика данных:

- `training_data/<date>/rect/` - нормальные изображения, разложенные по датам/папкам печати.
- `training_data/<date>/script_masks/` и `training_data/<date>/correlation_script_masks/` - маски объектов печати для нормальных изображений.
- `anomalous_data/` - изображения с дефектами. Для них сейчас нет масок дефектов.

Маски в `training_data` являются масками объектов, а не масками аномалий. Они нужны, чтобы выделить область печати, нарезать нормальные объекты на фрагменты и отфильтровать почти пустой фон.

Сопоставление изображения и маски выполняется внутри одной даты по имени/номеру файла. Если подходящей маски нет или найдено несколько подходящих масок, изображение пропускается.

## Подготовленная выборка 3D-принтера

Путь по умолчанию: `datasets/processed_printer_dataset/`.

Эта папка генерируется скриптом и не должна храниться в GitHub. Базовый запуск:

```bash
python code/prepare_dataset.py --mode all
```

Запуск с явными параметрами:

```bash
python code/prepare_dataset.py --mode all --input_dir datasets/3d_printer_dataset --output_dir datasets/processed_printer_dataset --image_size 320 --min_non_background_ratio 0.2
```

Основные параметры:

- `--input_dir`, `--dataset_dir`, `--data_dir` - корень датасета с папками `training_data/` и `anomalous_data/`.
- `--output_dir` - папка для результата.
- `--image_size` - итоговый размер квадратных фрагментов. По умолчанию `320`.
- `--step_size` - шаг нарезки больших изображений. По умолчанию половина `--image_size`.
- `--min_non_background_ratio`, `--min_object_ratio` - минимальная доля маски/не фоновых пикселей во фрагменте. По умолчанию `0.2`.

Ожидаемая структура результата:

```text
processed_printer_dataset/
  training/
    <date>/
      objects_parts/
      objects_parts_masks/
      full_objects/
      full_objects_masks/
  anomalies/
    objects_parts/
  anomalies_masked/
    objects_parts/
    objects_parts_masks/
    full_objects/
    full_objects_masks/
```

Семантика результата:

- `training/<date>/objects_parts/` - квадратные фрагменты нормальных изображений.
- `training/<date>/objects_parts_masks/` - бинарные маски объектов для этих фрагментов с теми же именами файлов.
- `training/<date>/full_objects/` - полные crop-объекты, если они проходят фильтр. Они не обязательно приведены к `--image_size`.
- `training/<date>/full_objects_masks/` - бинарные маски объектов для `full_objects/`.
- `anomalies/objects_parts/` - фрагменты аномальных изображений. Маски для них не создаются, потому что в исходном `anomalous_data/` нет разметки дефектов.
- `anomalies_masked/` - legacy-режим `--masked_anomalies`; здесь тоже сохраняются object masks, а не anomaly masks.

## Как адаптировать ноутбук MVTec AD под 3D-принтер

Ключевое отличие: в MVTec есть train/test и ground-truth маски дефектов, а в текущем датасете 3D-принтера есть нормальные изображения с object masks и аномальные изображения без anomaly masks.

При переносе ноутбука:

- `MVTecAD/<class>/train/good/` соответствует `processed_printer_dataset/training/*/objects_parts/`.
- `MVTecAD/<class>/test/<defect_type>/` соответствует `processed_printer_dataset/anomalies/objects_parts/`.
- `MVTecAD/<class>/ground_truth/<defect_type>/` прямого аналога сейчас не имеет.
- `processed_printer_dataset/training/*/objects_parts_masks/` можно использовать как object/foreground masks, но нельзя считать их разметкой дефектов.
- Pixel-level метрики на датасете 3D-принтера сейчас нельзя честно считать так же, как на MVTec, пока нет масок дефектов для `anomalous_data/`.
- Image-level метрики можно считать, если нормальные фрагменты брать из `training/*/objects_parts/`, а аномальные - из `anomalies/objects_parts/`.
- Если модель ожидает MVTec-подобный объект класса, для 3D-принтера лучше завести один pseudo-class, например `printer`, и внутри него использовать подготовленные `training/` и `anomalies/`.

## Текущий FastFlow split на данных принтера

Для сравнения FastFlow ResNet18 и DeiT используется
`datasets/processed_printer_dataset_384/`. Файлы в этой папке не перемещаются
между физическими train/calibration/test-каталогами: принадлежность к выборке
задаёт manifest
`experiments/printer/dataset_v384_audit/printer_split_v2_final.csv`.

Версия split: `printer_384_v2_final`. Состав:

| Split | Good | Anomalies | Независимые исходные съёмки |
|---|---:|---:|---:|
| train | 1000 | 0 | 212 |
| normal_val_loss | 160 | 0 | 44 |
| calibration | 40 | 21 | 5 good + 5 anomalous |
| test | 53 | 45 | 9 good + 7 anomalous |

Основная единица разделения нормальных данных — исходное изображение, а не
тайл. Для аномалий единица разделения — исходная съёмка. Между split нет
пересечений по исходной съёмке, object group и SHA-256. `normal_val_loss`
используется только для выбора checkpoint по normal likelihood; labeled
`calibration` — для top-k и порога; `test` остаётся закрытым до фиксации этих
параметров.

Папки `_review_split_*` внутри локального датасета — только визуальные копии
для ручной проверки и не читаются обучающим pipeline. Оставшиеся допустимые
отклонения в `good` намеренно сохранены как hard negatives. Решения ручной
проверки записаны в
`experiments/printer/dataset_v384_audit/printer_split_v2_final_domain_review.md`.

## Legacy: filtered printer dataset

Путь: `datasets/filtered_printer_dataset/`.

Эта папка оставлена для старого сценария `--masked_anomalies`. В этом режиме скрипт сравнивает `3d_printer_dataset/training_data` с `filtered_printer_dataset/training_data` и обрабатывает отсутствующие во filtered кадры как `anomalies_masked`.

Если этот режим не нужен, для основной подготовки датасета достаточно `datasets/3d_printer_dataset/`.
