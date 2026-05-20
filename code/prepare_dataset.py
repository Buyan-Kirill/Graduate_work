import os
import numpy as np
from PIL import Image
import argparse
import warnings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings("ignore", module="PIL")

DEFAULT_DATASET_DIR = "datasets/3d_printer_dataset"
DEFAULT_OUTPUT_DIR = "datasets/processed_printer_dataset"
DEFAULT_FILTERED_DIR = "datasets/filtered_printer_dataset"
DEFAULT_IMAGE_SIZE = 320
DEFAULT_MIN_OBJECT_RATIO = 0.2
MASK_FOLDERS = ["script_masks", "correlation_script_masks"]


def progress_bar(iterable, **kwargs):
    if tqdm is None:
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("ascii", " #")
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(iterable, **kwargs)


def log_message(message):
    if tqdm is None:
        print(message)
    else:
        tqdm.write(message)


def is_image_file(filename):
    return (
        not filename.startswith("._")
        and filename.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def find_mask_path(date_path, img_file):
    """Находит маску для изображения по номеру внутри папки даты"""
    img_stem = os.path.splitext(img_file)[0]
    candidates = []

    for mask_folder in MASK_FOLDERS:
        mask_path = os.path.join(date_path, mask_folder)
        if not os.path.isdir(mask_path):
            continue

        for mask_file in sorted(os.listdir(mask_path)):
            if not is_image_file(mask_file):
                continue

            mask_stem = os.path.splitext(mask_file)[0]
            if mask_file == img_file or mask_stem == img_stem:
                candidates.append(os.path.join(mask_path, mask_file))

    if len(candidates) == 1:
        return candidates[0], None

    if len(candidates) == 0:
        return None, "missing_masks"

    return None, "ambiguous_masks"


def load_and_normalize_image(path):
    """Загружает и нормализует изображение для корректного отображения"""
    img = Image.open(path)
    
    # Если изображение 16-битное, нормализуем его
    if img.mode == 'I;16' or img.mode == 'I':
        img_array = np.array(img)
        
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        
        if max_val > min_val:
            img_array = (img_array - min_val) * (255 / (max_val - min_val))
        
        img = Image.fromarray(img_array.astype(np.uint8))
    
    return img.convert("RGB")


def calculate_non_black_ratio(image, threshold=10):
    """Вычисляет долю не черных пикселей в изображении"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    if len(img_array.shape) == 3:
        non_black_mask = np.any(img_array > threshold, axis=2)
        non_black_count = np.count_nonzero(non_black_mask)
    else:
        non_black_count = np.count_nonzero(img_array > threshold)
    
    return non_black_count / total_pixels if total_pixels > 0 else 0


def save_binary_mask(mask, path):
    """Сохраняет бинарную маску объекта"""
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img.save(path)


def _process_training_image(img, mask, img_file, output_dir, 
                           target_size, step_size, padding, min_object_ratio, save_full_objects,
                           stats=None, mask_background=False):
    """Обрабатывает тренировочное изображение с маской"""
    if img.size[::-1] != mask.shape[:2]:
        log_message(f"  Пропущено: размеры изображения и маски не совпадают ({img_file})")
        if stats is not None:
            stats["size_mismatch"] += 1
        return 0, 0
    
    unique_vals = np.unique(mask)
    unique_vals = unique_vals[unique_vals != 0]
    
    processed_count = 0
    full_objects_count = 0
    
    for obj_val in unique_vals:
        if stats is not None:
            stats["objects_total"] += 1

        obj_mask = (mask == obj_val)

        if mask_background:
            obj_img = np.array(img)
            for c in range(3):
                obj_img[:, :, c] *= obj_mask.astype(np.uint8)
            obj_img = Image.fromarray(obj_img.astype(np.uint8))
        else:
            obj_img = img

        y_indices, x_indices = np.where(obj_mask)
        if len(x_indices) == 0:
            continue
            
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        min_x_pad = max(0, min_x - padding)
        min_y_pad = max(0, min_y - padding)
        max_x_pad = min(mask.shape[1], max_x + 1 + padding)
        max_y_pad = min(mask.shape[0], max_y + 1 + padding)
        
        bbox_width = max_x_pad - min_x_pad
        bbox_height = max_y_pad - min_y_pad
        
        if save_full_objects:
            full_mask = obj_mask[min_y_pad:max_y_pad, min_x_pad:max_x_pad]
            obj_pixels = np.count_nonzero(full_mask)
            total_pixels = full_mask.size
            
            if obj_pixels / total_pixels >= min_object_ratio:
                full_obj_img = obj_img.crop((min_x_pad, min_y_pad, max_x_pad, max_y_pad))
                full_obj_name = f"{img_file[:-4]}_obj{obj_val}_full.png"
                full_obj_path = os.path.join(output_dir, "full_objects", full_obj_name)
                full_obj_img.save(full_obj_path)

                full_mask_path = os.path.join(output_dir, "full_objects_masks", full_obj_name)
                save_binary_mask(full_mask, full_mask_path)
                full_objects_count += 1
        
        # Если объект слишком маленький, расширяем до минимального размера
        if bbox_width < target_size[0] or bbox_height < target_size[1]:
            center_x = (min_x_pad + max_x_pad) // 2
            center_y = (min_y_pad + max_y_pad) // 2
            
            min_x_pad = max(0, center_x - target_size[0]//2)
            max_x_pad = min(mask.shape[1], center_x + target_size[0]//2)
            min_y_pad = max(0, center_y - target_size[1]//2)
            max_y_pad = min(mask.shape[0], center_y + target_size[1]//2)
            
            bbox_width = max_x_pad - min_x_pad
            bbox_height = max_y_pad - min_y_pad
        
        if bbox_width > target_size[0] or bbox_height > target_size[1]:
            # Объект слишком большой - делим на части с заданным шагом
            for y in range(min_y_pad, max_y_pad, step_size[1]):
                for x in range(min_x_pad, max_x_pad, step_size[0]):
                    x1, y1 = x, y
                    x2 = min(x + target_size[0], max_x_pad)
                    y2 = min(y + target_size[1], max_y_pad)
                    
                    if x2 - x1 < target_size[0] * 0.5 or y2 - y1 < target_size[1] * 0.5:
                        continue
                    
                    tile_mask = obj_mask[y1:y2, x1:x2]
                    obj_pixels = np.count_nonzero(tile_mask)
                    total_pixels = tile_mask.size
                    
                    if obj_pixels / total_pixels < min_object_ratio:
                        if stats is not None:
                            stats["objects_rejected_min_ratio"] += 1
                        continue
                    
                    tile = obj_img.crop((x1, y1, x2, y2))
                    tile_mask_img = Image.fromarray((tile_mask.astype(np.uint8) * 255))

                    if tile.size != target_size:
                        padded = Image.new("RGB", target_size, (0, 0, 0))
                        padded.paste(tile, (0, 0))
                        tile = padded

                        padded_mask = Image.new("L", target_size, 0)
                        padded_mask.paste(tile_mask_img, (0, 0))
                        tile_mask_img = padded_mask

                    output_name = os.path.join("objects_parts", f"{img_file[:-4]}_obj{obj_val}_tile_{x1}_{y1}.png")
                    tile.save(os.path.join(output_dir, output_name))

                    mask_output_name = os.path.join("objects_parts_masks", f"{img_file[:-4]}_obj{obj_val}_tile_{x1}_{y1}.png")
                    tile_mask_img.save(os.path.join(output_dir, mask_output_name))
                    processed_count += 1
        else:
            # Объект помещается целиком
            bbox = (min_x_pad, min_y_pad, max_x_pad, max_y_pad)
            
            bbox_mask = obj_mask[min_y_pad:max_y_pad, min_x_pad:max_x_pad]
            obj_pixels = np.count_nonzero(bbox_mask)
            total_pixels = bbox_mask.size
            
            if obj_pixels / total_pixels < min_object_ratio:
                if stats is not None:
                    stats["objects_rejected_min_ratio"] += 1
                continue
            
            cropped = obj_img.crop(bbox)
            
            ratio = min(target_size[0]/bbox_width, target_size[1]/bbox_height)
            new_size = (int(bbox_width * ratio), int(bbox_height * ratio))
            resized = cropped.resize(new_size, Image.LANCZOS)
            resized_mask = Image.fromarray((bbox_mask.astype(np.uint8) * 255)).resize(new_size, Image.NEAREST)

            centered = Image.new("RGB", target_size, (0, 0, 0))
            centered_mask = Image.new("L", target_size, 0)
            pos = ((target_size[0] - new_size[0]) // 2,
                   (target_size[1] - new_size[1]) // 2)
            centered.paste(resized, pos)
            centered_mask.paste(resized_mask, pos)

            output_name = os.path.join("objects_parts", f"{img_file[:-4]}_obj{obj_val}_centered.png")
            centered.save(os.path.join(output_dir, output_name))

            mask_output_name = os.path.join("objects_parts_masks", f"{img_file[:-4]}_obj{obj_val}_centered.png")
            centered_mask.save(os.path.join(output_dir, mask_output_name))
            processed_count += 1
            
    return processed_count, full_objects_count


def _process_image_pair(img_path, mask_path_file, img_file, output_dir, 
                       target_size, step_size, padding, min_object_ratio, save_full_objects,
                       stats=None, mask_background=False):
    """Обрабатывает пару изображение-маска из файлов"""
    img = load_and_normalize_image(img_path)
    mask = np.array(Image.open(mask_path_file))
    
    return _process_training_image(
        img, mask, img_file, output_dir,
        target_size, step_size, padding, min_object_ratio, save_full_objects, stats, mask_background
    )


def _process_date_folder(date_path, date_folder, output_dir, target_size, step_size, 
                        padding, min_object_ratio, save_full_objects, mask_background=False):
    """Обрабатывает папку с датой для тренировочных данных"""
    rect_path = os.path.join(date_path, "rect")
    mask_paths = []

    for mask_folder in MASK_FOLDERS:
        candidate = os.path.join(date_path, mask_folder)
        if os.path.isdir(candidate):
            mask_paths.append(candidate)

    if not os.path.isdir(rect_path) or not mask_paths:
        return 0, 0

    output_dir_with_date = os.path.join(output_dir, date_folder)
    os.makedirs(output_dir_with_date, exist_ok=True)
    os.makedirs(os.path.join(output_dir_with_date, "objects_parts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_with_date, "objects_parts_masks"), exist_ok=True)
    if save_full_objects:
        os.makedirs(os.path.join(output_dir_with_date, "full_objects"), exist_ok=True)
        os.makedirs(os.path.join(output_dir_with_date, "full_objects_masks"), exist_ok=True)
    
    processed_count = 0
    full_objects_count = 0
    stats = {
        "images_total": 0,
        "images_processed": 0,
        "missing_masks": 0,
        "ambiguous_masks": 0,
        "size_mismatch": 0,
        "objects_total": 0,
        "objects_rejected_min_ratio": 0,
        "errors": 0,
    }
    
    image_files = [f for f in sorted(os.listdir(rect_path)) if is_image_file(f)]
    for img_file in progress_bar(image_files, desc=f"{date_folder}", leave=False):
        stats["images_total"] += 1
            
        img_path = os.path.join(rect_path, img_file)
        mask_path_file, mask_status = find_mask_path(date_path, img_file)
        
        if mask_path_file is None:
            stats[mask_status] += 1
            continue
            
        try:
            count, full_count = _process_image_pair(
                img_path, mask_path_file, img_file, output_dir_with_date,
                target_size, step_size, padding, min_object_ratio, save_full_objects,
                stats, mask_background
            )
            processed_count += count
            full_objects_count += full_count
            stats["images_processed"] += 1
            
        except Exception as e:
            stats["errors"] += 1
            log_message(f"  Ошибка при обработке {img_file}: {str(e)}")

    log_message(
        f"{date_folder}: "
        f"кадры {stats['images_processed']}/{stats['images_total']} | "
        f"без маски {stats['missing_masks']} | "
        f"неоднозначных масок {stats['ambiguous_masks']} | "
        f"объекты {stats['objects_total']} | "
        f"отброшено фрагм. {stats['objects_rejected_min_ratio']} | "
        f"сохранено фрагм. {processed_count} | "
        f"сохранено целиком {full_objects_count} | "
        f"ошибки {stats['errors']}"
    )
    
    return processed_count, full_objects_count


def _process_anomalous_image(img, img_file, output_dir, target_size,
                            step_size, min_object_ratio):
    """Обрабатывает аномальное изображение (без маски)"""
    processed_count = 0
    full_objects_count = 0
    
    img_width, img_height = img.size
    
    # Определяем bounding box всего изображения с отступом
    min_x = 0
    min_y = 0
    max_x = img_width
    max_y = img_height
    
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # Если изображение слишком маленькое, расширяем до минимального размера
    if bbox_width < target_size[0] or bbox_height < target_size[1]:
        center_x = img_width // 2
        center_y = img_height // 2
        
        min_x = max(0, center_x - target_size[0]//2)
        max_x = min(img_width, center_x + target_size[0]//2)
        min_y = max(0, center_y - target_size[1]//2)
        max_y = min(img_height, center_y + target_size[1]//2)
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
    
    # Обрабатываем изображение
    if bbox_width > target_size[0] or bbox_height > target_size[1]:
        # Изображение слишком большое - делим на части с заданным шагом
        for y in range(min_y, max_y, step_size[1]):
            for x in range(min_x, max_x, step_size[0]):
                x1, y1 = x, y
                x2 = min(x + target_size[0], max_x)
                y2 = min(y + target_size[1], max_y)
                
                if x2 - x1 < target_size[0] * 0.5 or y2 - y1 < target_size[1] * 0.5:
                    continue
                
                tile = img.crop((x1, y1, x2, y2))
                non_black_ratio = calculate_non_black_ratio(tile)
                
                if non_black_ratio < min_object_ratio:
                    continue
                
                if tile.size != target_size:
                    padded = Image.new("RGB", target_size, (0, 0, 0))
                    padded.paste(tile, (0, 0))
                    tile = padded
                
                output_name = os.path.join("objects_parts", f"{img_file[:-4]}_tile_{x1}_{y1}.png")
                tile.save(os.path.join(output_dir, output_name))
                processed_count += 1
    else:
        # Изображение помещается целиком
        bbox = (min_x, min_y, max_x, max_y)
        
        cropped = img.crop(bbox)
        non_black_ratio = calculate_non_black_ratio(cropped)
        
        if non_black_ratio < min_object_ratio:
            return processed_count, full_objects_count
        
        ratio = min(target_size[0]/bbox_width, target_size[1]/bbox_height)
        new_size = (int(bbox_width * ratio), int(bbox_height * ratio))
        resized = cropped.resize(new_size, Image.LANCZOS)
        
        centered = Image.new("RGB", target_size, (0, 0, 0))
        pos = ((target_size[0] - new_size[0]) // 2,
               (target_size[1] - new_size[1]) // 2)
        centered.paste(resized, pos)
        
        output_name = os.path.join("objects_parts", f"{img_file[:-4]}_centered.png")
        centered.save(os.path.join(output_dir, output_name))
        processed_count += 1
        
    return processed_count, full_objects_count


def _process_anomalous_folder(base_dir, output_dir, target_size, step_size, 
                             padding, min_object_ratio, save_full_objects):
    """Обрабатывает папку с аномальными данными (прямые изображения)"""
    log_message(f"Обрабатываем аномальные данные из: {base_dir}")

    os.makedirs(os.path.join(output_dir, "objects_parts"), exist_ok=True)
    if save_full_objects:
        os.makedirs(os.path.join(output_dir, "full_objects"), exist_ok=True)
    
    processed_count = 0
    full_objects_count = 0
    images_total = 0
    images_processed = 0
    errors = 0
    
    image_files = [f for f in sorted(os.listdir(base_dir)) if is_image_file(f)]
    for img_file in progress_bar(image_files, desc="anomalous", leave=False):
        images_total += 1
            
        img_path = os.path.join(base_dir, img_file)
        
        try:
            img = load_and_normalize_image(img_path)
            
            count, full_count = _process_anomalous_image(
                img, img_file, output_dir,
                target_size, step_size, min_object_ratio
            )
            processed_count += count
            full_objects_count += full_count
            images_processed += 1
            
        except Exception as e:
            errors += 1
            log_message(f"  Ошибка при обработке {img_file}: {str(e)}")

    log_message(
        f"  Кадров: {images_processed}/{images_total}, "
        f"фрагментов: {processed_count}, ошибок: {errors}"
    )
    
    return processed_count, full_objects_count


def process_data(
    base_dir,
    output_dir,
    target_size=(320, 320),
    step_size=None,
    padding=10,
    min_object_ratio=0.2,
    save_full_objects=True,
    mask_background=False,
    data_type="training"
):
    """
    Обрабатывает датасет, создавая изображения фиксированного размера с объектами или их частями.
    """
    os.makedirs(output_dir, exist_ok=True)

    if step_size is None:
        step_size = (max(1, target_size[0] // 2), max(1, target_size[1] // 2))
    
    log_message(f"Начинаем обработку {data_type} данных. Результаты будут сохранены в: {output_dir}")
    
    total_processed = 0
    total_full_objects = 0
    
    if data_type == "training":
        date_folders = [
            d for d in sorted(os.listdir(base_dir))
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        for date_folder in progress_bar(date_folders, desc="dates", leave=True):
            date_path = os.path.join(base_dir, date_folder)
                
            processed_count, full_objects_count = _process_date_folder(
                date_path, date_folder, output_dir, target_size, step_size, 
                padding, min_object_ratio, save_full_objects, mask_background
            )
            total_processed += processed_count
            total_full_objects += full_objects_count
            
    elif data_type == "anomalous":
        processed_count, full_objects_count = _process_anomalous_folder(
            base_dir, output_dir, target_size, step_size, 
            padding, min_object_ratio, save_full_objects
        )
        total_processed += processed_count
        total_full_objects += full_objects_count
    
    log_message(f"\nОбработка {data_type} данных завершена!")
    log_message(f"Создано фрагментов: {total_processed}")
    if save_full_objects:
        log_message(f"Сохранено полных объектов: {total_full_objects}")


def process_training_data(
    dataset_dir=DEFAULT_DATASET_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    target_size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
    step_size=None,
    min_object_ratio=DEFAULT_MIN_OBJECT_RATIO
):
    """Обрабатывает только тренировочные данные"""
    process_data(
        base_dir=os.path.join(dataset_dir, "training_data"),
        output_dir=os.path.join(output_dir, "training"),
        target_size=target_size,
        step_size=step_size,
        padding=10,
        min_object_ratio=min_object_ratio,
        save_full_objects=True,
        mask_background=False,
        data_type="training"
    )


def process_anomalous_data(
    dataset_dir=DEFAULT_DATASET_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    target_size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
    step_size=None,
    min_object_ratio=DEFAULT_MIN_OBJECT_RATIO
):
    """Обрабатывает только аномальные данные"""
    process_data(
        base_dir=os.path.join(dataset_dir, "anomalous_data"),
        output_dir=os.path.join(output_dir, "anomalies"),
        target_size=target_size,
        step_size=step_size,
        padding=0,
        min_object_ratio=min_object_ratio,
        save_full_objects=False,
        mask_background=False,
        data_type="anomalous"
    )


def process_masked_anomalies(
    dataset_dir=DEFAULT_DATASET_DIR,
    filtered_dir=DEFAULT_FILTERED_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    target_size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
    step_size=None,
    min_object_ratio=DEFAULT_MIN_OBJECT_RATIO
):
    """Обрабатывает изображения, которые есть в оригинальных 3d_printer_dataset/training_data, но отсутствуют в
    filtered_dataset/training_data для получения аномальных изображений, для которых есть маска"""

    source_base_dir = os.path.join(dataset_dir, "training_data")
    filtered_base_dir = os.path.join(filtered_dir, "training_data")
    output_dir = os.path.join(output_dir, "anomalies_masked")

    os.makedirs(os.path.join(output_dir, "objects_parts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "objects_parts_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "full_objects"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "full_objects_masks"), exist_ok=True)

    total_processed = 0
    total_full_objects = 0
    total_missing_masks = 0
    total_ambiguous_masks = 0

    if not os.path.isdir(source_base_dir):
        log_message(f"Источник masked anomalies не найден: {source_base_dir}")
        return
    if not os.path.isdir(filtered_base_dir):
        log_message(f"Filtered dataset не найден: {filtered_base_dir}")
        return

    filtered_by_date = {}
    for date_folder in sorted(os.listdir(filtered_base_dir)):
        date_path = os.path.join(filtered_base_dir, date_folder)
        if not os.path.isdir(date_path):
            continue

        rect_path = os.path.join(date_path, "rect")
        if not os.path.isdir(rect_path):
            continue

        filtered_files_in_date = set()
        for f in sorted(os.listdir(rect_path)):
            if is_image_file(f):
                filtered_files_in_date.add(f)

        filtered_by_date[date_folder] = filtered_files_in_date

    log_message(f"Найдено {len(filtered_by_date)} дат в filtered_dataset/training_data")

    source_date_folders = [
        d for d in sorted(os.listdir(source_base_dir))
        if os.path.isdir(os.path.join(source_base_dir, d))
    ]
    for date_folder in progress_bar(source_date_folders, desc="masked dates", leave=True):
        date_path = os.path.join(source_base_dir, date_folder)

        rect_path = os.path.join(date_path, "rect")
        mask_paths = []
        for mask_folder in MASK_FOLDERS:
            candidate = os.path.join(date_path, mask_folder)
            if os.path.isdir(candidate):
                mask_paths.append(candidate)

        if not os.path.isdir(rect_path) or not mask_paths:
            continue

        image_files = [f for f in sorted(os.listdir(rect_path)) if is_image_file(f)]
        for img_file in progress_bar(image_files, desc=f"{date_folder}", leave=False):

            if date_folder in filtered_by_date and img_file in filtered_by_date[date_folder]:
                continue

            img_path = os.path.join(rect_path, img_file)
            mask_path_file, mask_status = find_mask_path(date_path, img_file)
            if mask_path_file is None:
                if mask_status == "missing_masks":
                    total_missing_masks += 1
                elif mask_status == "ambiguous_masks":
                    total_ambiguous_masks += 1
                continue

            try:
                img = load_and_normalize_image(img_path)
                mask = np.array(Image.open(mask_path_file))

                count, full_count = _process_training_image(
                    img, mask, img_file, output_dir,
                    target_size=target_size,
                    step_size=step_size,
                    padding=10,
                    min_object_ratio=min_object_ratio,
                    save_full_objects=True,
                    mask_background=False
                )
                total_processed += count
                total_full_objects += full_count

            except Exception as e:
                log_message(f"  Ошибка при обработке {img_file} (masked anomaly): {str(e)}")

    log_message(f"\nОбработка masked anomalies завершена!")
    log_message(f"Создано фрагментов: {total_processed}")
    log_message(f"Сохранено полных объектов: {total_full_objects}")
    log_message(f"Без маски: {total_missing_masks}, неоднозначных масок: {total_ambiguous_masks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обработка датасета для обучения и аномальных данных')
    parser.add_argument('--mode', type=str, choices=['training', 'anomalous', 'all'], default='all',
                    help='Режим обработки: training (только тренировочные), anomalous (только аномальные), all (все)')
    parser.add_argument('--dataset_dir', '--data_dir', '--input_dir', type=str, default=DEFAULT_DATASET_DIR,
                    help='Путь к датасету с папками training_data и anomalous_data')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                    help='Путь для сохранения подготовленного датасета')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                    help='Итоговый размер квадратного изображения')
    parser.add_argument('--step_size', type=int, default=None,
                    help='Шаг нарезки больших изображений на фрагменты. По умолчанию половина --image_size')
    parser.add_argument('--min_object_ratio', '--min_non_background_ratio', type=float,
                    default=DEFAULT_MIN_OBJECT_RATIO,
                    help='Минимальная доля маски/не фоновых пикселей во фрагменте')
    parser.add_argument('--filtered_dir', type=str, default=DEFAULT_FILTERED_DIR,
                    help='Путь к filtered dataset для legacy-режима --masked_anomalies')
    parser.add_argument('--masked_anomalies', action='store_true',
                    help='Legacy: обрабатывать кадры из training_data, отсутствующие в --filtered_dir/training_data, как аномалии с масками')

    args = parser.parse_args()

    if args.image_size <= 0:
        parser.error('--image_size должен быть больше 0')
    if args.step_size is not None and args.step_size <= 0:
        parser.error('--step_size должен быть больше 0')
    if args.min_object_ratio < 0 or args.min_object_ratio > 1:
        parser.error('--min_object_ratio должен быть в диапазоне [0, 1]')

    target_size = (args.image_size, args.image_size)
    step_size = None
    if args.step_size is not None:
        step_size = (args.step_size, args.step_size)
    
    if args.mode == 'training' or args.mode == 'all':
        log_message("=== ОБРАБОТКА ТРЕНИРОВОЧНЫХ ДАННЫХ ===")
        process_training_data(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            target_size=target_size,
            step_size=step_size,
            min_object_ratio=args.min_object_ratio
        )
    
    if args.mode == 'anomalous' or args.mode == 'all':
        log_message("\n=== ОБРАБОТКА АНОМАЛЬНЫХ ДАННЫХ ===")
        process_anomalous_data(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            target_size=target_size,
            step_size=step_size,
            min_object_ratio=args.min_object_ratio
        )
    
    if args.masked_anomalies:
        log_message("\n=== ОБРАБОТКА MASKED ANOMALIES ===")
        process_masked_anomalies(
            dataset_dir=args.dataset_dir,
            filtered_dir=args.filtered_dir,
            output_dir=args.output_dir,
            target_size=target_size,
            step_size=step_size,
            min_object_ratio=args.min_object_ratio
        )
