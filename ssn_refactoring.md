# Запрос для LLM: рефакторинг SuperSimpleNet notebooks

Нужно аккуратно отрефакторить notebooks с SuperSimpleNet в проекте:

- `code/SuperSimpleNet_mvtec.ipynb`
- `code/SuperSimpleNet_printer.ipynb`
- `code/SuperSimpleNet_printer_head_finetune.ipynb`
- при необходимости свериться с `code/ssn_calculator.ipynb`

Контекст: проект обучает SuperSimpleNet для anomaly detection/localization. Нужно сохранить текущую экспериментальную логику, но убрать неоднозначные места, привести inference/training path к реальному API `anomalib` и сделать визуализацию/метрики менее обманчивыми.

## Что уже проверено

В `code/ssn_calculator.ipynb` был пошагово проверен один пример через обученную SSN-модель.

Правильный inference path:

```python
features = model.feature_extractor(img)
adapted = model.adaptor(features)
anomaly_logits_lowres, image_logit = model.segdec(adapted)
anomaly_logits = model.anomaly_map_generator(
    anomaly_logits_lowres,
    final_size=img.shape[-2:],
)
anomaly_prob = torch.sigmoid(anomaly_logits)
```

Проверка показала:

```text
Max |model.anomaly_map - generator_pred_map|: 0.0
Max |model.pred_score - module_pred_score|: 0.0
```

Значит `model(img).anomaly_map` уже проходит через `model.anomaly_map_generator(...)` и уже имеет размер входа. Нельзя заменять этот шаг простым `torch.nn.functional.interpolate`, если цель - вручную воспроизвести `model.forward()`.

Training path тоже проверен: в `model.train()` `model.anomaly_generator(...)` стохастически создает synthetic anomalies. В `model.eval()` генератор может вернуть нулевые labels/masks, поэтому диагностировать training path нужно только в train-mode.

## Обязательные изменения

1. Заменить `predict_with_augmentations` на явный `predict_anomaly_map`.

   В текущих notebooks функция называется как test-time augmentation, но часто фактически не делает augmentations или содержит закомментированные flips. Это вводит в заблуждение.

   Нужная функция:

   ```python
   def predict_anomaly_map(model, img, apply_sigmoid=True):
       model.eval()
       with torch.no_grad():
           output = model(img)
           anomaly_map = output.anomaly_map
           if apply_sigmoid:
               anomaly_map = torch.sigmoid(anomaly_map)
       return anomaly_map
   ```

   Если нужен backward compatibility, оставить:

   ```python
   def predict_with_augmentations(model, img, true_img_size=None):
       anomaly_map = predict_anomaly_map(model, img, apply_sigmoid=True)
       if true_img_size is not None and anomaly_map.shape[-2:] != tuple(true_img_size):
           anomaly_map = postprocess_anomaly_map(anomaly_map, true_img_size)
       return anomaly_map
   ```

   Но новый код должен использовать `predict_anomaly_map`.

2. Не делать повторный resize `output.anomaly_map`, если размер уже правильный.

   Для `SupersimplenetModel.forward()` в inference:

   ```python
   anomaly_map, anomaly_score = self.segdec(adapted)
   anomaly_map = self.anomaly_map_generator(anomaly_map, final_size=output_size)
   return InferenceBatch(anomaly_map=anomaly_map, pred_score=anomaly_score)
   ```

   Поэтому `output.anomaly_map` уже приведена к размеру входа. Для MVTec, если вход `256x256`, карта уже `256x256`.

   `postprocess_anomaly_map(...)` нужен только если нужно привести карту к original image/mask size для pixel metric против original-size mask. Но если mask тоже была transformed to `256x256`, повторный postprocess не нужен.

3. Оставить `torch.sigmoid(output.anomaly_map)`, но сделать его явным и единым.

   `output.anomaly_map` - raw logits. Для визуализации и pixel-level metrics нужна probability-like карта:

   ```python
   anomaly_prob = torch.sigmoid(output.anomaly_map)
   ```

   Не смешивать подходы `+ 1`, raw logits и sigmoid в разных местах notebooks.

4. Не применять sigmoid к image score для ROC AUC, если метрика называется raw.

   `output.pred_score` - image-level logit. Для `roc_auc_score` можно использовать raw logit, потому что sigmoid монотонен. Если сохраняется sigmoid score для CSV/визуализации, явно назвать его `image_score_sigmoid`.

   Рекомендуемые имена:

   - `image_roc_auc_raw`
   - `raw_pred_score`
   - `sigmoid_pred_score`

5. Унифицировать training loss path.

   Сейчас train и validation loss могут использовать разные пути:

   - train: `model(images, masks=dummy_masks, labels=dummy_labels)`
   - validation: ручной путь `feature_extractor -> adaptor -> anomaly_generator -> segdec`

   Нужно вынести единый helper:

   ```python
   def ssn_training_forward(model, images, masks=None, labels=None):
       if masks is None:
           masks = torch.zeros((images.shape[0], images.shape[-2], images.shape[-1]), device=images.device)
       if labels is None:
           labels = torch.zeros((images.shape[0],), device=images.device)

       features = model.feature_extractor(images)
       adapted = model.adaptor(features)
       target_mask = model.downsample_mask(masks, *features.shape[-2:])
       target_label = labels.to(torch.float32)
       train_features, target_mask, target_label = model.anomaly_generator(adapted, target_mask, target_label)
       pred_map, pred_score = model.segdec(train_features)
       return pred_map, pred_score, target_mask, target_label
   ```

   Использовать его и в train loss, и в validation loss.

6. Диагностировать training path только в `model.train()`.

   `model.anomaly_generator` стохастический и должен проверяться в train-mode. Если прогонять его после `model.eval()`, можно ошибочно решить, что synthetic anomalies не генерируются.

7. Исправить визуализацию anomaly maps.

   Проблема: matplotlib без `vmin/vmax` растягивает colormap по min/max конкретной картинки. Из-за этого normal image может выглядеть ярко-аномальным.

   Требование:

   ```python
   imshow(anomaly_map, cmap="jet", vmin=0.0, vmax=1.0)
   ```

   Также обязательно:

   - denormalize input image;
   - `torch.clamp(image, 0, 1)`;
   - не показывать raw logits как probability map без подписи.

8. Добавить foreground/ROI diagnostic или foreground mask.

   Обнаружено: на normal hazelnut яркое пятно может появляться далеко от объекта. Это не обязательно ошибка forward path; SSN считает карту по всему изображению и не знает, что фон вне объекта нужно игнорировать.

   Нужно добавить один из вариантов:

   - foreground mask по яркости/segmentation;
   - crop/ROI объекта перед моделью;
   - умножение anomaly map на foreground mask только для визуализации/ROI metric;
   - явный подсчет inside/outside object max/mean.

   Важно: foreground mask не должна ломать метрики, если benchmark предполагает full image. Для MVTec можно оставить full-image метрики, но визуально показывать masked overlay как diagnostic.

9. Исправить mutable default arguments.

   Заменить:

   ```python
   test_classes=[]
   ```

   на:

   ```python
   test_classes=None
   test_classes = set(test_classes or [])
   ```

10. Сохранять `state_dict` наряду с pickle-моделью.

    Сейчас используется `torch.save(model, "trained_model.pth")`, что зависит от версии `anomalib` и Python pickle. Для совместимости можно оставить, но добавить:

    ```python
    torch.save(model.state_dict(), "trained_model_weights.pth")
    ```

    В идеале грузить так:

    ```python
    model = SupersimplenetModel(...)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    ```

11. Уточнить `final_metrics.txt`.

    Сейчас файл может писать последние значения метрик, хотя сохраненная модель может быть best model по early stopping. Нужно либо:

    - сохранять `best_epoch` и `best_<metric>`;
    - либо после загрузки best state заново прогонять validation/test и писать метрики именно best model.

12. Добавить CUDA/kernel guard в начало notebooks.

    Пользователь запускает из WSL/kernel `my_env`. Нужно явно проверять:

    ```python
    import sys, torch
    print(sys.executable)
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    assert "+cu128" in torch.__version__
    assert torch.cuda.is_available()
    ```

13. Не делать TTA без валидации.

    Flips могут быть невалидны для некоторых категорий/датасетов, особенно если orientation важна. Если TTA возвращается, нужно:

    - явно назвать функцию `predict_with_tta`;
    - применять sigmoid одинаково ко всем branches;
    - инвертировать transforms корректно;
    - сравнить метрики с/без TTA.

14. Проверить mask transforms.

    Для масок использовать nearest interpolation:

    ```python
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
    ```

    Не использовать bilinear для binary masks.

15. Убрать лишние/неиспользуемые imports и диагностические ячейки.

    Например `cv2`, `measure`, `average_precision_score`, `Adam`, `AdamW`, scheduler imports могут быть не везде нужны. Но не делать большой косметический рефактор, если он мешает проверке эксперимента.

## Критерии приемки

1. `code/ssn_calculator.ipynb` воспроизводит inference path:

   ```text
   Max |model.anomaly_map - generator_pred_map|: 0.0
   Max |model.pred_score - module_pred_score|: 0.0
   ```

2. Training diagnostic в `model.train()` показывает, что `anomaly_generator` стохастически создает ненулевые masks/labels.

3. В основных notebooks:

   - нет псевдо-TTA под именем `predict_with_augmentations`;
   - `output.anomaly_map` не resize'ится повторно без необходимости;
   - sigmoid применяется централизованно;
   - visualizations используют `vmin=0, vmax=1`;
   - mask resize использует nearest;
   - CUDA guard присутствует;
   - `state_dict` сохраняется.

4. Метрики до/после рефакторинга должны быть сопоставимы. Если они изменились, нужно явно объяснить почему: например удалили повторный resize, поменяли TTA, changed mask size, changed foreground masking.

