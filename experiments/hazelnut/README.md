# Hazelnut

MVTec `hazelnut`: проверка гипотез для SuperSimpleNet перед переносом на датасет 3D-принтера.

## Содержимое

- `v3_new_scheduler/` - SuperSimpleNet с новым scheduler.
- `mvtec_head_finetune/` - основной head fine-tuning эксперимент.
- `mvtec_head_finetune_copy/` - повтор/вариант fine-tuning эксперимента.

## Что сравнивалось

- до fine-tuning;
- после fine-tuning classification head;
- после fine-tuning segmentation decoder;
- weak/top-k MIL варианты.

Опорный результат `v3_new_scheduler`: `Pixels_ROC_AUC = 0.9839`, `Classification_ROC_AUC = 0.9942`.
