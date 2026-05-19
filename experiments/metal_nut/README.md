# Metal Nut

MVTec `metal_nut`: ранняя проверка SuperSimpleNet, scheduler и качества pixel-level локализации.

## Содержимое

- `v1`, `v2`, `v3` - последовательные версии эксперимента.
- `v3_laptop` - повтор/перенос запуска на другой машине.
- `v3_new_scheduler` - проверка нового scheduler.

Лучшие сохраненные результаты: `v3` и `v3_laptop`, оба с `Classification_ROC_AUC = 1.0`.


## try_6

- Run with updated `code/SuperSimpleNet_mvtec.ipynb`: normal train/val for synthetic val loss, labeled val/test metrics, synthetic fallback for missing labels, final per-class metrics, and one-per-class visualizations.
