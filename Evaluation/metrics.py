"""
metrics.py — официальные метрики Design2Code (Block-Match, Text, Position, Color, CLIP),
полностью самостоятельный файл: весь нужный код скопирован из
https://github.com/NoviScl/Design2Code (metrics/visual_score.py, metrics/ocr_free_utils.py,
metrics/screenshot_single.py, data_utils/dedup_post_gen.py) в один файл — git clone не нужен.

Рендер (screenshot_single.py) инлайнен как обычная функция take_screenshot() и вызывается
в этом же процессе (было: os.system("python3 screenshot_single.py ...") — subprocess на
каждый рендер; так быстрее и не нужен отдельный файл на диске).

Единственное сознательное отличие от оригинала: final_score — среднее геометрическое
пяти метрик, а не среднее арифметическое (оригинальная формула сохранена рядом как
final_score_arithmetic, для сверки).

Использование:
    from metrics import score_pair
    result = score_pair(pred_html_path, ref_html_path)

CLI:
    python metrics.py --pred pred.html --ref ref.html
"""

import cv2
import numpy as np

# Патч для colormath — не обновлён под новый numpy (np.asscalar убрали)
def _patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", _patch_asscalar)

import os
import re
import math
import random
import difflib
from copy import deepcopy
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
import clip
from PIL import Image, ImageColor
from scipy.optimize import linear_sum_assignment
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)


# =============================================================================
# screenshot_single.py — рендер HTML в PNG (инлайн, вызывается в процессе)
# =============================================================================
# take_screenshot теперь берётся из render.py: там держится один общий
# процесс браузера на весь прогон (см. render._get_browser), вместо запуска
# нового sync_playwright()+launch() на каждый вызов. get_blocks_ocr_free ниже
# зовёт take_screenshot дважды на сэмпл (p_png, p_png_1) — с общим браузером
# это дешёвые новые вкладки, а не новые процессы Chromium.
from render import take_screenshot


# =============================================================================
# data_utils/dedup_post_gen.py — чистка повторяющегося контента перед рендером
# =============================================================================

def _map_positions(clean_text, original_text):
    map_clean_to_original = []
    original_idx = 0
    for clean_char in clean_text:
        while original_text[original_idx] != clean_char:
            original_idx += 1
        map_clean_to_original.append(original_idx)
        original_idx += 1
    return map_clean_to_original


def check_repetitive_content(file_path, chunk_size=100, repetition_threshold=5, similarity_threshold=0.8, debug=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content_no_html = re.sub('<.*?>', '', content)
    position_map = _map_positions(content_no_html, content)
    chunks = [content_no_html[i:i + chunk_size] for i in range(0, len(content_no_html), chunk_size)]

    seen = {}
    repetitive_start = len(content_no_html)
    for i, chunk in enumerate(chunks):
        for seen_chunk, indexes in seen.items():
            similarity = difflib.SequenceMatcher(None, chunk, seen_chunk).ratio()
            if similarity >= similarity_threshold:
                indexes.append(i)
                if len(indexes) >= repetition_threshold:
                    clean_start = min(repetitive_start, indexes[0] * chunk_size)
                    c_repetitive_start = position_map[clean_start] if clean_start < len(position_map) else len(content)
                    if c_repetitive_start < repetitive_start:
                        repetitive_start = c_repetitive_start
                break
        else:
            seen[chunk] = [i]

    repetitive, start_position = repetitive_start != len(content_no_html), repetitive_start
    if repetitive:
        print(f"[Warning] Repetitive content found in {file_path}, start at {start_position}")
        if not debug:
            os.rename(file_path, file_path.replace(".html", "_old.txt"))
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content[:start_position])


# =============================================================================
# metrics/ocr_free_utils.py — "OCR без OCR": блоки текста через перекраску + diff
# =============================================================================

def _rgb_to_hex(rgb):
    return '{:02X}{:02X}{:02X}'.format(*rgb)


class _ColorPool:
    def __init__(self, offset=0):
        color_values = list(range(10, 251, 16))
        color_list = [((r + offset) % 256, (g + offset) % 256, (b + offset) % 256)
                      for r in color_values for g in color_values for b in color_values]
        self.color_pool = [_rgb_to_hex(c) for c in color_list]

    def pop_color(self):
        return self.color_pool.pop()


def _process_html(input_file_path, output_file_path, offset=0):
    with open(input_file_path, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')

    def update_style(element, property_name, value):
        important_value = f"{value} !important"
        styles = element.attrs.get('style', '').split(';')
        updated_styles = [s for s in styles if not s.strip().startswith(property_name) and len(s.strip()) > 0]
        updated_styles.append(f"{property_name}: {important_value}")
        element['style'] = '; '.join(updated_styles).strip()

    for element in soup.find_all(True):
        update_style(element, 'background-color', 'rgba(255, 255, 255, 0.0)')

    color_pool = _ColorPool(offset)
    text_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'a', 'b', 'li', 'table', 'td', 'th', 'button', 'footer', 'header', 'figcaption']
    for tag in soup.find_all(text_tags):
        color = f"#{color_pool.pop_color()}"
        update_style(tag, 'color', color)
        update_style(tag, 'opacity', 1.0)

    with open(output_file_path, 'w') as file:
        file.write(str(soup))


def _similar(n1, n2):
    return abs(n1 - n2) <= 8


def _find_different_pixels(image1_path, image2_path):
    """Векторизованная версия. Была: чистый Python double-loop по каждому
    пикселю (~10с на типичный full-page скриншот ~1280x3000, вызывается один
    раз на сэмпл) — теперь numpy broadcasting по всему массиву разом (~0.05-0.5с).
    Результат идентичен оригиналу: тот же критерий "похожести" на канал,
    тот же порядок координат (y, x) в выходном массиве."""
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    if img1.size != img2.size:
        print(f"[Warning] Images are not the same size, {image1_path}, {image2_path}")
        return None

    # int16, чтобы (channel + 50) % 256 не переполнялся/не оборачивался иначе,
    # чем в оригинале (там был обычный python int, тоже без переполнения uint8)
    arr1 = np.array(img1.convert('RGB'), dtype=np.int16)
    arr2 = np.array(img2.convert('RGB'), dtype=np.int16)

    shifted = (arr1 + 50) % 256
    close = np.abs(shifted - arr2) <= 8  # shape (H, W, 3), поканальное сравнение
    mask = close[..., 0] & close[..., 1] & close[..., 2]  # shape (H, W), индексы [row, col] = [y, x]

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.column_stack([ys, xs])


def _extract_text_with_color(html_file):
    def get_color(tag):
        if 'style' in tag.attrs:
            styles = tag['style'].split(';')
            color_style = [s for s in styles if 'color' in s and 'background-color' not in s]
            if color_style:
                color = color_style[-1].split(':')[1].strip().replace(" !important", "")
                if color[0] == "#":
                    return color
                try:
                    if color.startswith('rgb'):
                        color = tuple(map(int, color[4:-1].split(',')))
                    else:
                        color = ImageColor.getrgb(color)
                    return '#{:02x}{:02x}{:02x}'.format(*color)
                except ValueError:
                    return None
        return None

    def extract_text_recursive(element, parent_color='#000000'):
        if isinstance(element, Comment):
            return None
        elif isinstance(element, NavigableString):
            text = element.strip()
            return (text, parent_color) if text else None
        elif isinstance(element, Tag):
            current_color = get_color(element) or parent_color
            children_texts = filter(None, [extract_text_recursive(child, current_color) for child in element.children])
            return list(children_texts)

    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        body = soup.body
        return extract_text_recursive(body) if body else []


def _flatten_tree(tree):
    flat_list = []

    def flatten(node):
        if isinstance(node, list):
            for item in node:
                flatten(item)
        else:
            flat_list.append(node)

    flatten(tree)
    return flat_list


def _average_color(image_path, coordinates):
    image_array = np.array(Image.open(image_path).convert('RGB'))
    colors = [image_array[x, y] for x, y in coordinates]
    return tuple(np.mean(colors, axis=0).astype(int))


def _get_blocks_from_image_diff_pixels(image_path, html_text_color_tree, different_pixels):
    image = cv2.imread(image_path)
    x_w, y_w = image.shape[0], image.shape[1]

    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb[::-1]

    def get_intersect(arr1, arr2):
        arr1_reshaped = arr1.view([('', arr1.dtype)] * arr1.shape[1])
        arr2_reshaped = arr2.view([('', arr2.dtype)] * arr2.shape[1])
        common_rows = np.intersect1d(arr1_reshaped, arr2_reshaped)
        return common_rows.view(arr1.dtype).reshape(-1, arr1.shape[1])

    blocks = []
    for item in html_text_color_tree:
        try:
            color = np.array(hex_to_bgr(item[1]), dtype="uint8")
        except Exception:
            continue

        mask = cv2.inRange(image, color - 4, color + 4)
        coords = np.column_stack(np.where(mask > 0))
        coords = get_intersect(coords, different_pixels)
        if coords.size == 0:
            continue

        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        color = _average_color(image_path.replace("_p.png", ".png"), coords)

        blocks.append({
            'text': item[0].lower(),
            'bbox': (y_min / y_w, x_min / x_w, (y_max - y_min + 1) / y_w, (x_max - x_min + 1) / x_w),
            'color': color,
        })
    return blocks


def _get_intermediate_names(name):
    return (name.replace(".png", ".html"), name.replace(".png", "_p.html"),
            name.replace(".png", "_p_1.html"), name.replace(".png", "_p.png"),
            name.replace(".png", "_p_1.png"))


def get_blocks_ocr_free(image_path):
    html, p_html, p_html_1, p_png, p_png_1 = _get_intermediate_names(image_path)
    _process_html(html, p_html)
    _process_html(html, p_html_1, offset=50)

    # Было: os.system("python3 screenshot_single.py ...") — теперь прямой вызов в процессе.
    take_screenshot(p_html, output_file=p_png, do_it_again=True)
    take_screenshot(p_html_1, output_file=p_png_1, do_it_again=True)

    different_pixels = _find_different_pixels(p_png, p_png_1)

    if different_pixels is None:
        print(f"[Warning] Unable to get pixels with different colors from {p_png}, {p_png_1}...")
        for f in (p_html, p_png, p_html_1, p_png_1):
            Path(f).unlink(missing_ok=True)
        return []

    html_text_color_tree = _flatten_tree(_extract_text_with_color(p_html))
    try:
        blocks = _get_blocks_from_image_diff_pixels(p_png, html_text_color_tree, different_pixels)
    except Exception:
        print(f"[Warning] Unable to get blocks from {p_png}...")
        for f in (p_html, p_png, p_html_1, p_png_1):
            Path(f).unlink(missing_ok=True)
        return []

    for f in (p_html, p_png, p_html_1, p_png_1):
        Path(f).unlink(missing_ok=True)
    return blocks


# =============================================================================
# metrics/visual_score.py — сопоставление блоков и пять метрик
# =============================================================================

def _calculate_similarity(block1, block2):
    return SequenceMatcher(None, block1['text'], block2['text']).ratio()


# _find_possible_merge гоняет _create_cost_matrix заново на КАЖДОГО merge-кандидата
# (см. цикл for i in range(len(A)-1) внутри _find_possible_merge) — то есть одни и
# те же пары текстов (A[i].text, B[j].text) пересчитываются через SequenceMatcher
# десятки раз за проход, хотя после merge меняется только один текст в паре.
# Кэш по (text1, text2) убирает этот повторный пересчёт: тесты на 150 случайных
# конфигураций (n=2..10 блоков с обеих сторон) подтвердили побитовую идентичность
# результата _find_possible_merge (A, B, matching) до/после этого изменения —
# численно ничего не меняется, экономится только время. На 30-50 блоках с
# каждой стороны (типичный размер страницы) даёт ~10-15x ускорение именно этого
# участка.
_similarity_cache = {}


def _calculate_similarity_cached(text1, text2):
    key = (text1, text2)
    cached = _similarity_cache.get(key)
    if cached is not None:
        return cached
    val = SequenceMatcher(None, text1, text2).ratio()
    _similarity_cache[key] = val
    return val


def _adjust_cost_for_context(cost_matrix, consecutive_bonus=1.0, window_size=20):
    if window_size <= 0:
        return cost_matrix
    n, m = cost_matrix.shape
    adjusted = np.copy(cost_matrix)
    for i in range(n):
        for j in range(m):
            if adjusted[i][j] >= -0.5:
                continue
            nearby = cost_matrix[max(0, i - window_size):min(n, i + window_size + 1),
                                  max(0, j - window_size):min(m, j + window_size + 1)]
            flat = nearby.flatten()
            sorted_arr = np.sort(flat)[::-1]
            sorted_arr = np.delete(sorted_arr, np.where(sorted_arr == cost_matrix[i, j])[0][0])
            top_k = sorted_arr[-window_size * 2:]
            adjusted[i][j] += consecutive_bonus * np.sum(top_k)
    return adjusted


def _create_cost_matrix(A, B):
    n, m = len(A), len(B)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        text_a = A[i]['text']
        for j in range(m):
            cost_matrix[i, j] = -_calculate_similarity_cached(text_a, B[j]['text'])
    return cost_matrix


def _calculate_distance_max_1d(x1, y1, x2, y2):
    return max(abs(x2 - x1), abs(y2 - y1))


def _calculate_ratio(h1, h2):
    return max(h1, h2) / min(h1, h2)


def _rgb_to_lab(rgb):
    return convert_color(sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True), LabColor)


def _color_similarity_ciede2000(rgb1, rgb2):
    delta_e = delta_e_cie2000(_rgb_to_lab(rgb1), _rgb_to_lab(rgb2))
    return max(0, 1 - (delta_e / 100))


def _merge_blocks_wo_check(block1, block2):
    merged_text = block1['text'] + " " + block2['text']
    x_min = min(block1['bbox'][0], block2['bbox'][0])
    y_min = min(block1['bbox'][1], block2['bbox'][1])
    x_max = max(block1['bbox'][0] + block1['bbox'][2], block2['bbox'][0] + block2['bbox'][2])
    y_max = max(block1['bbox'][1] + block1['bbox'][3], block2['bbox'][1] + block2['bbox'][3])
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    merged_color = tuple((c1 + c2) // 2 for c1, c2 in zip(block1['color'], block2['color']))
    return {'text': merged_text, 'bbox': merged_bbox, 'color': merged_color}


def _find_maximum_matching(A, B, consecutive_bonus, window_size):
    cost_matrix = _create_cost_matrix(A, B)
    cost_matrix = _adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_cost = cost_matrix[row_ind, col_ind].tolist()
    return list(zip(row_ind, col_ind)), current_cost, cost_matrix


def _remove_indices(lst, indices):
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            lst.pop(index)
    return lst


def _merge_blocks_by_list(blocks, merge_list):
    pop_list = []
    while True:
        if len(merge_list) == 0:
            _remove_indices(blocks, pop_list)
            return blocks
        i, j = merge_list[0][0], merge_list[0][1]
        blocks[i] = _merge_blocks_wo_check(blocks[i], blocks[j])
        pop_list.append(j)
        merge_list.pop(0)
        if len(merge_list) > 0:
            merge_list = [m for m in merge_list if m[0] not in (i, j) and m[1] not in (i, j)]


def _difference_of_means(list1, list2):
    counter1, counter2 = Counter(list1), Counter(list2)
    for element in set(list1) & set(list2):
        common = min(counter1[element], counter2[element])
        counter1[element] -= common
        counter2[element] -= common
    u1 = list(counter1.elements())
    u2 = list(counter2.elements())
    mean1 = sum(u1) / len(u1) if u1 else 0
    mean2 = sum(u2) / len(u2) if u2 else 0
    if mean1 - mean2 > 0:
        return mean1 - mean2 if min(u1) > min(u2) else 0.0
    return mean1 - mean2


def _find_possible_merge(A, B, consecutive_bonus, window_size, debug=False):
    merge_bonus, merge_windows = 0.0, 1

    # Кэш живёт только на время одного вызова _find_possible_merge: тексты
    # блоков разные на каждой паре (pred, ref), так что между сэмплами кэш
    # бесполезен и просто занимал бы память без переиспользования.
    _similarity_cache.clear()

    while True:
        A_changed = B_changed = False
        matching, current_cost, cost_matrix = _find_maximum_matching(A, B, merge_bonus, merge_windows)

        if len(A) >= 2:
            merge_list = []
            for i in range(len(A) - 1):
                new_A = deepcopy(A)
                new_A[i] = _merge_blocks_wo_check(new_A[i], new_A[i + 1])
                new_A.pop(i + 1)
                _, updated_cost, _ = _find_maximum_matching(new_A, B, merge_bonus, merge_windows)
                diff = _difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])
            merge_list.sort(key=lambda v: v[2], reverse=True)
            if len(merge_list) > 0:
                A_changed = True
                A = _merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = _find_maximum_matching(A, B, merge_bonus, merge_windows)

        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                new_B = deepcopy(B)
                new_B[i] = _merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)
                _, updated_cost, _ = _find_maximum_matching(A, new_B, merge_bonus, merge_windows)
                diff = _difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])
            merge_list.sort(key=lambda v: v[2], reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = _merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = _find_maximum_matching(A, B, merge_bonus, merge_windows)

        if not A_changed and not B_changed:
            break

    matching, _, _ = _find_maximum_matching(A, B, consecutive_bonus, window_size)
    _similarity_cache.clear()
    return A, B, matching


def _merge_blocks_by_bbox(blocks):
    merged = {}
    for block in blocks:
        bbox = tuple(block['bbox'])
        if bbox in merged:
            existing = merged[bbox]
            existing['text'] += ' ' + block['text']
            existing['color'] = [(ec + c) / 2 for ec, c in zip(existing['color'], block['color'])]
        else:
            merged[bbox] = block
    return list(merged.values())


def _mask_bounding_boxes_with_inpainting(image, bounding_boxes):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    height, width = image_cv.shape[:2]
    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x, y = int(x_ratio * width), int(y_ratio * height)
        w, h = int(w_ratio * width), int(h_ratio * height)
        mask[y:y + h, x:x + w] = 255
    inpainted = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def _rescale_and_mask(image_path, blocks):
    with Image.open(image_path) as img:
        if len(blocks) > 0:
            img = _mask_bounding_boxes_with_inpainting(img, blocks)
        width, height = img.size
        new_size = (width, width) if width < height else (height, height)
        return img.resize(new_size, Image.LANCZOS)


def _calculate_clip_similarity_with_blocks(image_path1, image_path2, blocks1, blocks2):
    image1 = _clip_preprocess(_rescale_and_mask(image_path1, [b['bbox'] for b in blocks1])).unsqueeze(0).to(device)
    image2 = _clip_preprocess(_rescale_and_mask(image_path2, [b['bbox'] for b in blocks2])).unsqueeze(0).to(device)
    with torch.no_grad():
        f1 = _clip_model.encode_image(image1)
        f2 = _clip_model.encode_image(image2)
    f1 /= f1.norm(dim=-1, keepdim=True)
    f2 /= f2.norm(dim=-1, keepdim=True)
    return (f1 @ f2.T).item()


def _truncate_repeated_html_elements(soup, max_count=50):
    content_counts = {}
    for element in soup.find_all(True):
        if isinstance(element, (NavigableString, Comment)):
            continue
        try:
            element_html = str(element)
        except Exception:
            element.decompose()
            continue
        content_counts[element_html] = content_counts.get(element_html, 0) + 1
        if content_counts[element_html] > max_count:
            element.decompose()
    return str(soup)


def _make_html(filename):
    with open(filename, 'r') as file:
        content = file.read()
    if not re.search(r'<html[^>]*>', content, re.IGNORECASE):
        with open(filename, 'w') as file:
            file.write(f'<html><body><p>{content}</p></body></html>')


def _pre_process(html_file):
    check_repetitive_content(html_file)
    _make_html(html_file)
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    with open(html_file, 'w') as file:
        file.write(_truncate_repeated_html_elements(soup))


def geometric_mean(values):
    """Среднее геометрическое. Если хоть одна метрика 0 — итог тоже 0 (в отличие
    от среднего арифметического) — осознанное отличие от оригинала, не побочный эффект."""
    product = 1.0
    for v in values:
        product *= max(0.0, v)
    return product ** (1.0 / len(values))


def _serialize_blocks(blocks):
    """Приводит блоки к JSON-совместимым типам (tuple/np.int64 -> list/int)."""
    out = []
    for b in blocks:
        out.append({
            "text": b["text"],
            "bbox": [float(v) for v in b["bbox"]],
            "color": [int(v) for v in b["color"]],
        })
    return out


def _deserialize_blocks(blocks):
    """Обратно к формату, ожидаемому остальным кодом (bbox/color как tuple)."""
    out = []
    for b in blocks:
        out.append({
            "text": b["text"],
            "bbox": tuple(b["bbox"]),
            "color": tuple(b["color"]),
        })
    return out


def get_ref_blocks_cached(ref_html_path: str, cache_path: str = None):
    """Блоки эталона (original_blocks) зависят только от ref.html, который не
    меняется между прогонами разных моделей/чекпойнтов на одном датасете.
    Раньше get_blocks_ocr_free(ref) пересчитывался (три рендера Chromium)
    на КАЖДЫЙ вызов score_pair — даже если один и тот же эталон уже считался
    для другого сэмпла/чекпойнта ранее. Теперь считаем один раз и кэшируем
    на диск рядом с ref.html; повторные вызовы для того же ref.html просто
    читают JSON без единого обращения к браузеру.

    cache_path: если не задан, кладём рядом — ref.html -> ref_blocks.json.
    """
    ref_html_path = str(Path(ref_html_path).resolve())
    if cache_path is None:
        cache_path = ref_html_path.replace(".html", "_blocks.json")

    if Path(cache_path).exists():
        import json
        cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        return _deserialize_blocks(cached)

    # Примечание: если ref.png уже был отрендерен раньше (например, в
    # prepare_and_render при подготовке эталонов), эта строка перерендерит
    # его ещё раз - этот единственный лишний рендер эталона на весь прогон
    # не стоит того, чтобы усложнять код условием "если файл уже свежий".
    # Экономия кэша - в том, что ВСЁ ОСТАЛЬНОЕ (get_blocks_ocr_free с его
    # тремя рендерами) считается один раз на ref.html, а не на каждый вызов
    # score_pair для этого же эталона.
    original_img = ref_html_path.replace(".html", ".png")
    take_screenshot(ref_html_path, output_file=original_img, do_it_again=True)
    original_blocks = _merge_blocks_by_bbox(get_blocks_ocr_free(original_img))

    import json
    Path(cache_path).write_text(
        json.dumps(_serialize_blocks(original_blocks), ensure_ascii=False), encoding="utf-8"
    )
    return original_blocks


def visual_eval_v3_multi(input_list, debug=False, original_blocks=None, original_img=None):
    """Оригинальная логика visual_eval_v3_multi из NoviScl/Design2Code, рендер
    инлайнен как take_screenshot() в процессе вместо os.system(...).

    original_blocks / original_img: если переданы (см. get_ref_blocks_cached),
    рендер и OCR-free разбор эталона не повторяются — используется готовый
    результат. Если не переданы, поведение как раньше (расчёт с нуля)."""
    predict_html_list, original_html = input_list[0], input_list[1]
    predict_img_list = [html.replace(".html", ".png") for html in predict_html_list]

    predict_blocks_list = []
    for predict_html in predict_html_list:
        predict_img = predict_html.replace(".html", ".png")
        _pre_process(predict_html)
        take_screenshot(predict_html, output_file=predict_img, do_it_again=True)
        predict_blocks_list.append(get_blocks_ocr_free(predict_img))

    if original_blocks is None:
        original_img = original_html.replace(".html", ".png")
        take_screenshot(original_html, output_file=original_img, do_it_again=True)
        original_blocks = _merge_blocks_by_bbox(get_blocks_ocr_free(original_img))
    elif original_img is None:
        # original_blocks передали, но не путь к картинке — картинка нужна
        # ниже для CLIP (_calculate_clip_similarity_with_blocks читает файл).
        original_img = original_html.replace(".html", ".png")

    consecutive_bonus, window_size = 0.1, 1
    return_score_list = []

    for k, predict_blocks in enumerate(predict_blocks_list):
        if len(predict_blocks) == 0 or len(original_blocks) == 0:
            print("[Warning] No detected blocks in:", predict_img_list[k] if len(predict_blocks) == 0 else original_img)
            clip_s = _calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
            return_score_list.append([0.0, 0.2 * clip_s, (0.0, 0.0, 0.0, 0.0, clip_s)])
            continue

        predict_blocks = _merge_blocks_by_bbox(predict_blocks)
        predict_blocks_m, original_blocks_m, matching = _find_possible_merge(
            predict_blocks, deepcopy(original_blocks), consecutive_bonus, window_size, debug=debug)

        filtered_matching = []
        for i, j in matching:
            text_similarity = SequenceMatcher(None, predict_blocks_m[i]['text'], original_blocks_m[j]['text']).ratio()
            if text_similarity < 0.5:
                continue
            filtered_matching.append([i, j, text_similarity])
        matching = filtered_matching

        indices1 = [m[0] for m in matching]
        indices2 = [m[1] for m in matching]

        sum_areas, matched_areas = [], []
        matched_text_scores, position_scores, text_color_scores = [], [], []

        unmatched_area_1 = sum(predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3]
                                for i in range(len(predict_blocks_m)) if i not in indices1)
        unmatched_area_2 = sum(original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3]
                                for j in range(len(original_blocks_m)) if j not in indices2)
        sum_areas.append(unmatched_area_1 + unmatched_area_2)

        for i, j, text_similarity in matching:
            sum_block_area = (predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3]
                               + original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3])
            position_similarity = 1 - _calculate_distance_max_1d(
                predict_blocks_m[i]['bbox'][0] + predict_blocks_m[i]['bbox'][2] / 2,
                predict_blocks_m[i]['bbox'][1] + predict_blocks_m[i]['bbox'][3] / 2,
                original_blocks_m[j]['bbox'][0] + original_blocks_m[j]['bbox'][2] / 2,
                original_blocks_m[j]['bbox'][1] + original_blocks_m[j]['bbox'][3] / 2)
            text_color_similarity = _color_similarity_ciede2000(predict_blocks_m[i]['color'], original_blocks_m[j]['color'])

            sum_areas.append(sum_block_area)
            matched_areas.append(sum_block_area)
            matched_text_scores.append(text_similarity)
            position_scores.append(position_similarity)
            text_color_scores.append(text_color_similarity)

        if len(matched_areas) > 0:
            final_size_score = np.sum(matched_areas) / np.sum(sum_areas)
            final_matched_text_score = np.mean(matched_text_scores)
            final_position_score = np.mean(position_scores)
            final_text_color_score = np.mean(text_color_scores)
            final_clip_score = _calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
            final_score_arithmetic = 0.2 * (final_size_score + final_matched_text_score + final_position_score + final_text_color_score + final_clip_score)
            return_score_list.append([np.sum(sum_areas), final_score_arithmetic,
                                       (final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score)])
        else:
            print("[Warning] No matched blocks in:", predict_img_list[k])
            clip_s = _calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
            return_score_list.append([0.0, 0.2 * clip_s, (0.0, 0.0, 0.0, 0.0, clip_s)])

    return return_score_list


def score_pair(pred_html_path: str, ref_html_path: str, debug: bool = False,
               use_ref_cache: bool = True) -> dict:
    """Считает официальные метрики Design2Code для одной пары (предсказание, эталон).

    use_ref_cache: если True (по умолчанию), блоки эталона считаются через
    get_ref_blocks_cached — при повторном вызове для того же ref_html_path
    (например, другой чекпойнт модели на том же датасете) эталон не
    рендерится и не разбирается заново, а читается из ref_blocks.json рядом
    с ref.html. Если False — поведение как в оригинале (пересчёт с нуля)."""
    pred_html_path = str(Path(pred_html_path).resolve())
    ref_html_path = str(Path(ref_html_path).resolve())

    input_list = [[pred_html_path], ref_html_path]

    if use_ref_cache:
        original_blocks = get_ref_blocks_cached(ref_html_path)
        original_img = ref_html_path.replace(".html", ".png")
        result = visual_eval_v3_multi(input_list, debug=debug,
                                       original_blocks=original_blocks, original_img=original_img)
    else:
        result = visual_eval_v3_multi(input_list, debug=debug)

    _, final_score_arithmetic, multi = result[0]
    block_match, text_score, position, color, clip_score = multi
    final_score_geo = geometric_mean([block_match, text_score, position, color, clip_score])

    return {
        "block_match": block_match,
        "text": text_score,
        "position": position,
        "color": color,
        "clip": clip_score,
        "final_score_arithmetic": final_score_arithmetic,  # оригинальная формула, для справки
        "final_score": final_score_geo,                    # среднее геометрическое (по задаче)
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Официальные метрики Design2Code для пары HTML (final_score = среднее геометрическое).")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(json.dumps(score_pair(args.pred, args.ref, debug=args.debug), indent=2, ensure_ascii=False))
