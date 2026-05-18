# Experiments

Результаты запусков по anomaly detection и локализации дефектов.

## Папки

- `metal_nut/` - MVTec `metal_nut`, проверка SuperSimpleNet и настроек обучения.
- `hazelnut/` - MVTec `hazelnut`, проверка SuperSimpleNet и head fine-tuning.
- `3D_printer_wide_resnet/` - ранние FastFlow/backbone-запуски с Wide ResNet.
- `3D_printer_resnet_18/` - ранние FastFlow/backbone-запуски с ResNet-18.
- `3D_printer_transformer/` - ранняя проверка transformer-подхода.
- `3D_printer_supersimplenet/` - базовый SuperSimpleNet на датасете 3D-принтера.
- `3D_printer_supersimplenet_head_finetune/` - текущие эксперименты SuperSimpleNet с fine-tuning.

## Типовые файлы

- `final_metrics.txt` - итоговые метрики запуска.
- `middle_logs/` - чекпоинты, промежуточные метрики, визуализации по эпохам.
- `trained_model*.pth` - сохраненные модели/веса.
- `*.png` - графики loss/metrics и примеры карт аномалий.
