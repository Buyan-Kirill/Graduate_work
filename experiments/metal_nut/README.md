# Metal Nut

MVTec `metal_nut`: ранняя проверка SuperSimpleNet, scheduler и качества pixel-level локализации.

## Содержимое

- `v1`, `v2`, `v3` - последовательные версии эксперимента.
- `v3_laptop` - повтор/перенос запуска на другой машине.
- `v3_new_scheduler` - проверка нового scheduler.

Лучшие сохраненные результаты: `v3` и `v3_laptop`, оба с `Classification_ROC_AUC = 1.0`.


## ssn_baseline

- Run with updated `code/SuperSimpleNet_mvtec.ipynb`: normal train/val for synthetic val loss, labeled val/test metrics, synthetic fallback for missing labels, final per-class metrics, and one-per-class visualizations.

## ssn_unsupervised_proxy_compare

- Notebook: `code/SuperSimpleNet_mvtec_unsupervised_compare.ipynb`.
- Purpose: compare fully unsupervised synthetic validation/test metrics against real labeled reference metrics.
- Split: 1 normal train dataset; validation and test each have fixed synthetic segmentation/classification references plus real segmentation/classification references.
- Synthetic references are generated once before training as fixed SSN synthetic masks/labels, then reused through the current model path.
- Important for future printer-part runs without masks: real labels are reference-only here; early stopping stays on unsupervised `Validation_Loss`.
