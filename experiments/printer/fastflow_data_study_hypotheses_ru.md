# FastFlow DeiT data study: полный реестр гипотез

## Статусы

| ID | Гипотеза | Статус | Проверяемые факты |
|---|---|---|---|
| D1 | Mild photometric augmentation устойчиво улучшает DeiT | Не подтверждена | Seed 42 primary `+0.017159`, object/source без изменения; повторов не было |
| D2 | Strong legacy augmentation ухудшает FastFlow из-за расширения normal distribution | Опровергнута как универсальное утверждение | Seed 42 primary `+0.075682`; object/source также выше |
| D3 | Strong augmentation устойчиво улучшает primary ranking относительно no-aug DeiT | Не подтверждена | Post-selection mean delta `-0.000170`; знаки `+/-`; fixed full map `-0.004602` |
| D4 | Strong augmentation снижает seed-вариативность DeiT | Поддержана описательно | Sample SD `0.010739` против `0.044221`, но только три seed |
| D5 | Strong DeiT не хуже ResNet18-384 | Поддержана на текущей calibration | Primary выше на 3/3 seed; post-selection mean delta `+0.021023`; независимого holdout нет |
| D6 | Найден единый стабильный top-k для strong DeiT | Не подтверждена | Выбраны `50%/50%/100%` map; full-map sensitivity сохраняет вывод только против ResNet |
| D7 | 1671 normal tiles лучше 1000 | Опровергнута для seed-42 recipe | Primary `0.888295` против `0.873182`: небольшой рост, но хуже 500 и strong; threshold хуже |
| D8 | Меньший date-balanced subset уменьшает вред повторов | Совместима с данными, не доказана | 500 tiles: `0.930909`, 1000: `0.873182`, 1671: `0.888295`; смешаны composition/steps/seed |
| D9 | Максимальное object/source coverage важнее date balance | Не подтверждена | Object-uniform 1000 `0.906591`, date-balanced 500 `0.930909` |
| D10 | Чем ниже normal-val loss, тем лучше anomaly ranking | Опровергнута между recipes | All-1671 loss `72784.56`, но AUC `0.888295`; strong loss выше, AUC лучше |
| D11 | Улучшение ranking гарантирует улучшение q95 threshold | Опровергнута | Seed 123 strong ranking немного выше, balanced accuracy ниже no-aug |
| D12 | Конкретная операция strong bundle полезна | Не проверена | Crop/flips/rotation/jitter/blur/erasing менялись одновременно |
| D13 | Дубли по датам являются установленной причиной деградации | Не доказана | Наблюдается 84.9% tiles в двух датах и немонотонность объёма, но нет equal-step paired repeats |

## D1. Mild photometric augmentation

Изменение было одно: `ColorJitter(brightness=0.05, contrast=0.05)` и
Gaussian blur с `p=0.15`.

На seed 42 primary AUC вырос с `0.873182` до `0.890341`, но object AUC
остался `0.925`, source AUC — `0.800`. Эффект мал относительно известной
seed-вариативности no-aug DeiT и не был выбран для повторов. Вывод:
перспективно, но доказательств устойчивого улучшения нет.

## D2-D4. Strong legacy augmentation

Bundle включает RandomResizedCrop, horizontal/vertical flips, rotation,
ColorJitter, частый blur и RandomErasing. На seed 42 он оказался лучшим из
пяти вариантов, что опровергло исходную гипотезу об обязательном вреде.

Post-selection результаты изменили интерпретацию:

| Seed | Strong - no-aug primary |
|---:|---:|
| 42, selection | +0.075682 |
| 123 | +0.004205 |
| 2025 | -0.004545 |

Большой прирост не реплицировался. При этом object AUC strong равен
`0.985/0.990/0.990`, source AUC — `0.880/0.920/0.920`; это стабильнее
no-aug. Возможная интерпретация: bundle действует как regularizer и уменьшает
variance, но не сдвигает ожидаемый tile ranking. Это гипотеза, а не
доказанный механизм.

## D5. Сравнение с ResNet18

Strong-minus-ResNet primary delta:

- seed 42: `+0.028977`;
- seed 123: `+0.022727`;
- seed 2025: `+0.019318`.

Object mean strong `0.988333` против `0.965000`; source mean одинаков
`0.906667`. Это достаточное основание считать strong DeiT кандидатом на
новый holdout. Это не основание объявлять production superiority:
cluster-bootstrap interval широк, calibration мала и участвовала в top-k
selection.

## D6. Top-k

Strong выбрал 73728 pixels на seed 42/123 и 147456 на seed 2025. На fixed
full map strong всё ещё выше ResNet18 на всех seed, но ниже no-aug DeiT на
обоих post-selection seed. Поэтому top-k `50%` нельзя замораживать как
универсальный оптимум только по двум попаданиям из трёх.

## D7-D9 и D13. Объём и состав train

Первый проход показывает немонотонность:

| Sampling | Tiles | Sources | Objects | Primary AUC |
|---|---:|---:|---:|---:|
| date-balanced | 500 | 167 | 397 | 0.930909 |
| date-balanced | 1000 | 212 | 687 | 0.873182 |
| all | 1671 | 235 | 800 | 0.888295 |
| object-uniform | 1000 | 235 | 800 | 0.906591 |

Факты совместимы с вредом повторяющихся tiles доминирующих дат, но не
изолируют его. При фиксированных 40 epochs число optimizer steps различается,
а все варианты выполнены на одном seed. Корректная будущая проверка требует
nested subsets, одинакового числа steps и минимум двух заранее заданных seed.

## D10-D11. Loss, ranking и threshold

Likelihood loss оценивает соответствие normal distribution и меняет масштаб
при изменении train distribution. Поэтому его абсолютные значения между
augmentation/data recipes не являются anomaly metric.

ROC AUC не использует threshold. q95-normal threshold фиксирует около 5% FPR
на calibration normals, но FNR зависит от score scale/separation. Поэтому
seed 123 может одновременно иметь лучший ranking strong и худшую balanced
accuracy, чем no-aug. В отчётах эти результаты должны оставаться раздельными.

## D12. Отдельные аугментации

Текущий эксперимент сравнил два отдельных режима (`mild` и весь `strong`
bundle), а не компоненты bundle. Нельзя приписать результат flip, crop,
blur или RandomErasing. Особенно RandomErasing и экстремальный crop могут
создавать нецелевые искусственные anomalies; переносить bundle в production
без нового holdout и component ablation рискованно.

## Что считается закрытым

- Нельзя объяснять качество только normal-val loss.
- Использование всех tiles не дало убедительного выигрыша.
- Большой seed-42 прирост strong augmentation не является устойчивой оценкой
  эффекта относительно no-aug.
- Strong DeiT является лучшим текущим кандидатом относительно ResNet18 на
  calibration, но требует нового независимого holdout.

## Что остаётся открытым

- Даст ли strong DeiT выигрыш на будущих независимых source groups.
- Какая операция strong bundle отвечает за снижение наблюдаемой variance.
- Поможет ли date-balanced 500 при equal optimizer steps и нескольких seed.
- Какой top-k переносится без повторной настройки на новую сессию.
