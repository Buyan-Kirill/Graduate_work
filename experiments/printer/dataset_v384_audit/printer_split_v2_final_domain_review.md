# Domain review: printer_384_v2_final

Status: approved for the planned FastFlow comparison.

The review was performed on the flat visual copy
`datasets/processed_printer_dataset_384/_review_split_v2_final_flat`.
Original dataset files were not moved, deleted, or modified.

Decisions:

- `anomalies/objects_parts/2025-08-08_14-09-52_L0128_1_tile_3072_960.png`
  contains an out-of-scope defect and is excluded from evaluation rather than
  relabeled as normal.
- Normal object groups `normal:2025-06-25_test:141_obj3` and
  `normal:2025-06-25_test:143_obj7` are visually ambiguous and are excluded as
  complete object groups.
- The remaining images are accepted in their assigned classes.
- Some accepted `good` images contain visible but permissible deviations. They
  are intentionally retained as in-scope hard negatives.

The exclusions above are encoded in `configs/printer_split_v2_final.json`.
