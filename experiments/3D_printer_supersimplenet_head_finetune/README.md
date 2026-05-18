# 3D Printer SuperSimpleNet Head Fine-Tune

Текущая линия экспериментов: SuperSimpleNet на датасете 3D-принтера с дополнительным fine-tuning голов модели.

## Содержимое

- `try_1` - `try_3` - отдельные попытки.
- `head_finetune_logs/` - логи fine-tuning головы.
- `final_eval*` - финальная оценка до/после fine-tuning.
- `head_finetune_comparison/` - сравнение состояний модели.
- `trained_model_head_finetuned*.pth` - модель после fine-tuning.

Задача серии - проверить, улучшает ли fine-tune качество обнаружения и локализации дефектов на данных 3D-принтера.
