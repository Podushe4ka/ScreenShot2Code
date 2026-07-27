#!/usr/bin/env python3
"""
view_arrow.py — просмотр содержимого .arrow файлов (Apache Arrow IPC),
корректно обрабатывая колонки с изображениями (Image), как их хранит
библиотека HuggingFace `datasets`.

Как datasets хранит картинки:
    Image-фича кодируется в Arrow как struct { bytes: binary, path: string }.
    Если печатать такой файл «в лоб», получаешь тысячи байт бинарщины в консоли.
    Этот скрипт распознаёт такие колонки и показывает вместо байтов
    краткую сводку (формат, размер, разрешение), а при желании — сохраняет
    картинки на диск.

Примеры:
    # схема + первые 5 строк
    python view_arrow.py data.arrow

    # первые 20 строк
    python view_arrow.py data.arrow -n 20

    # показать только выбранные колонки
    python view_arrow.py data.arrow -c text label

    # извлечь все изображения в папку ./images
    python view_arrow.py data.arrow --extract-images images

    # вывести всё в JSON (картинки заменяются описанием)
    python view_arrow.py data.arrow --json
"""

import argparse
import base64
import html as html_mod
import io
import json
import os
import sys

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except ImportError:
    sys.exit("Нужен pyarrow. Установите: pip install pyarrow")

# PIL опционален — без него мы всё равно покажем размер и формат по сигнатуре
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


# --------------------------------------------------------------------------
# Открытие файла: .arrow бывает в двух IPC-форматах — file и stream.
# datasets обычно пишет stream, но встречается и file. Пробуем оба.
# --------------------------------------------------------------------------
def open_arrow(path):
    with open(path, "rb") as f:
        data = f.read()
    # 1) IPC file format (начинается с магии b"ARROW1")
    try:
        reader = ipc.open_file(pa.BufferReader(data))
        return reader.read_all()
    except pa.lib.ArrowInvalid:
        pass
    # 2) IPC stream format
    try:
        reader = ipc.open_stream(pa.BufferReader(data))
        return reader.read_all()
    except pa.lib.ArrowInvalid as e:
        sys.exit(f"Не удалось прочитать как Arrow IPC (ни file, ни stream): {e}")


# --------------------------------------------------------------------------
# Распознавание «картиночных» колонок.
# --------------------------------------------------------------------------
def _struct_is_image(t):
    if pa.types.is_struct(t):
        names = {t.field(i).name for i in range(t.num_fields)}
        return {"bytes", "path"}.issubset(names)
    return False


def is_image_field(field):
    """Ловит и struct{bytes, path}, и list<struct{bytes, path}> — обе формы,
    в которых datasets хранит Image (одна картинка / список картинок)."""
    t = field.type
    if _struct_is_image(t):
        return True
    # list<item: struct{bytes, path}>
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _struct_is_image(t.value_type)
    return False


def image_columns(table):
    return [f.name for f in table.schema if is_image_field(f)]


# --------------------------------------------------------------------------
# Разбор одного значения-картинки.
# --------------------------------------------------------------------------
def describe_one_image(value):
    """value — dict {'bytes': b'...', 'path': '...'} или None."""
    if value is None:
        return "<None>"
    raw = value.get("bytes")
    path = value.get("path")
    if not raw:
        # картинка хранится ссылкой на файл, а не байтами
        return f"<Image path={path!r} (байты не встроены)>"

    n = len(raw)
    fmt, size = sniff_image(raw)
    parts = [f"<Image {n} bytes"]
    if fmt:
        parts.append(f"формат={fmt}")
    if size:
        parts.append(f"размер={size[0]}x{size[1]}")
    if path:
        parts.append(f"path={path!r}")
    return " ".join(parts) + ">"


def describe_image(value):
    """value может быть одной картинкой (dict), списком картинок (list) или None."""
    if value is None:
        return "<None>"
    if isinstance(value, list):
        if not value:
            return "[] (пустой список картинок)"
        items = [describe_one_image(v) for v in value]
        # чтобы не раздувать вывод при десятках картинок — показываем первые 3
        shown = items[:3]
        tail = f"  …ещё {len(items) - 3}" if len(items) > 3 else ""
        return f"[{len(items)} картинок]\n      " + "\n      ".join(shown) + tail
    return describe_one_image(value)


def sniff_image(raw):
    """Возвращает (формат, (w, h)). Через PIL, а если его нет — по сигнатуре."""
    if HAVE_PIL:
        try:
            with Image.open(io.BytesIO(raw)) as im:
                return im.format, im.size
        except Exception:
            return None, None
    # грубое определение формата по магическим байтам
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG", None
    if raw[:3] == b"\xff\xd8\xff":
        return "JPEG", None
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF", None
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "WEBP", None
    if raw[:2] == b"BM":
        return "BMP", None
    return None, None


def ext_for(raw, default=".bin"):
    fmt, _ = sniff_image(raw)
    return {
        "PNG": ".png", "JPEG": ".jpg", "GIF": ".gif",
        "WEBP": ".webp", "BMP": ".bmp",
    }.get(fmt, default)


# --------------------------------------------------------------------------
# Форматирование обычных (не картиночных) значений для читаемого вывода.
# --------------------------------------------------------------------------
def format_value(value, max_len=120):
    if isinstance(value, bytes):
        return f"<binary {len(value)} bytes>"
    s = repr(value)
    if len(s) > max_len:
        return s[:max_len] + f"… ({len(s)} симв.)"
    return s


# --------------------------------------------------------------------------
# Извлечение картинок на диск.
# --------------------------------------------------------------------------
def _save_one(raw, out_dir, base):
    if not raw:
        return 0
    with open(os.path.join(out_dir, base + ext_for(raw)), "wb") as f:
        f.write(raw)
    return 1


def extract_images(table, img_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for col in img_cols:
        column = table.column(col).to_pylist()
        for i, value in enumerate(column):
            if not value:
                continue
            if isinstance(value, list):
                # несколько картинок в строке — добавляем индекс j
                for j, item in enumerate(value):
                    if item:
                        saved += _save_one(item.get("bytes"), out_dir,
                                           f"{col}_{i:06d}_{j:02d}")
            else:
                saved += _save_one(value.get("bytes"), out_dir, f"{col}_{i:06d}")
    print(f"Сохранено изображений: {saved} → {out_dir}/")


# --------------------------------------------------------------------------
# Основной вывод.
# --------------------------------------------------------------------------
def print_schema(table, img_cols):
    print("=" * 70)
    print(f"Строк: {table.num_rows}   Колонок: {table.num_columns}")
    print("-" * 70)
    for field in table.schema:
        tag = "  [IMAGE]" if field.name in img_cols else ""
        print(f"  {field.name}: {field.type}{tag}")
    print("=" * 70)


def print_rows(table, n, cols, img_cols):
    cols = cols or table.schema.names
    limit = min(n, table.num_rows)
    for i in range(limit):
        print(f"\n─── строка {i} " + "─" * 50)
        for col in cols:
            if col not in table.schema.names:
                print(f"  (нет колонки {col!r})")
                continue
            value = table.column(col)[i].as_py()
            if col in img_cols:
                print(f"  {col}: {describe_image(value)}")
            else:
                print(f"  {col}: {format_value(value)}")


def dump_json(table, n, cols, img_cols):
    cols = cols or table.schema.names
    limit = min(n, table.num_rows) if n >= 0 else table.num_rows
    rows = []
    for i in range(limit):
        row = {}
        for col in cols:
            if col not in table.schema.names:
                continue
            value = table.column(col)[i].as_py()
            if col in img_cols:
                row[col] = describe_image(value)
            elif isinstance(value, bytes):
                row[col] = f"<binary {len(value)} bytes>"
            else:
                row[col] = value
        rows.append(row)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


# --------------------------------------------------------------------------
# HTML-отчёт: на каждую строку — картинка(и) + рендер и исходник HTML-колонок,
# чтобы визуально сопоставлять «скриншот ↔ код».
# --------------------------------------------------------------------------
def html_columns(table):
    """Строковые HTML-колонки (по имени), НЕПУСТЫЕ.
    Пустые (напр. current_html для drafting — ещё не используется) пропускаем."""
    cols = []
    for f in table.schema:
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type):
            if "html" in f.name.lower():
                if any((v.as_py() or "").strip() for v in table.column(f.name)):
                    cols.append(f.name)
    return cols


def _img_data_uri(item):
    raw = item.get("bytes") if item else None
    if not raw:
        return None
    fmt, _ = sniff_image(raw)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif",
            "WEBP": "image/webp", "BMP": "image/bmp"}.get(fmt, "image/png")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _row_images(value):
    """Возвращает список dict'ов картинок из ячейки (list или одиночная)."""
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def build_html_report(table, out_path, img_cols, html_cols, n):
    limit = min(n, table.num_rows) if n >= 0 else table.num_rows
    parts = ["""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Arrow report</title><style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f4f5;color:#18181b}
 header{padding:14px 20px;background:#18181b;color:#fff;position:sticky;top:0;z-index:5}
 .row{background:#fff;margin:18px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden}
 .row>h2{margin:0;padding:10px 16px;background:#e4e4e7;font-size:14px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
 .cell{padding:14px;border-top:1px solid #eee;min-width:0}
 .cell h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#71717a}
 /* Скриншот из датасета — это УЖЕ честный full-page рендер target_html (снят тем же
    render_full). Поэтому показываем сам скриншот рядом с кодом и НЕ ре-рендерим в iframe:
    браузерный iframe не расцепляет vh и высоту кадра (вьюпорт=высота элемента), поэтому
    воспроизвести full-page скрин при вьюпорте 1024 не может — прошлый iframe давал ложное
    расхождение (другой vh, обрезка, другой масштаб). Не возвращать iframe. */
 .shot{max-height:85vh;overflow:auto;border:1px solid #ddd;border-radius:6px;background:#fff}
 .shot img{display:block;max-width:100%}
 pre{max-height:85vh;overflow:auto;background:#0d1117;color:#c9d1d9;padding:12px;border-radius:6px;
     font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
 @media(max-width:820px){.grid{grid-template-columns:1fr}}
</style></head><body>"""]
    parts.append(f"<header>Arrow report — {html_mod.escape(os.path.basename(out_path))} · строк: {limit} "
                 f"· слева скриншот (= рендер target_html), справа код</header>")

    for i in range(limit):
        parts.append(f'<div class="row"><h2>Строка {i}</h2><div class="grid">')

        # скриншот(ы) — это и есть рендер target_html
        for col in img_cols:
            imgs = _row_images(table.column(col)[i].as_py())
            uris = [u for u in (_img_data_uri(x) for x in imgs) if u]
            parts.append('<div class="cell"><h3>' + html_mod.escape(col) + '</h3>')
            if uris:
                parts.append("".join(f'<div class="shot"><img src="{u}"></div>' for u in uris))
            else:
                parts.append("<em>нет картинок</em>")
            parts.append("</div>")

        # html-колонки: только исходник (рендер = скриншот слева)
        for col in html_cols:
            src = html_mod.escape(table.column(col)[i].as_py() or "")
            parts.append('<div class="cell"><h3>' + html_mod.escape(col) + ' (код)</h3>'
                         f'<pre>{src}</pre></div>')

        parts.append("</div></div>")

    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"HTML-отчёт готов: {out_path}  (строк: {limit})")


# --------------------------------------------------------------------------
# Сверка рендера: ре-рендер target_html тем же рендерером, что делал скриншот,
# и сравнение со скриншотом из датасета. Честная проверка «код рендерится так же»
# (в отличие от HTML-отчёта: там браузерный iframe не воспроизводит full-page).
# --------------------------------------------------------------------------
def verify_render(table, img_cols, out_dir, n):
    """Для n строк: рендерит target_html через convert_lib.render_full (ровно тот рендерер,
    которым снят скриншот) и сравнивает с картинкой из датасета — размеры + средняя попиксельная
    разница. Кладёт side-by-side [датасет | ре-рендер] в out_dir, помечает расхождения.
    Совпадение доказывает, что скриншот = честный рендер target_html. Нужны playwright +
    (target_html уже с precompiled Tailwind, второй раз компилировать не надо)."""
    if not HAVE_PIL:
        sys.exit("--verify нужен Pillow: pip install pillow")
    try:
        import convert_lib as C
        from PIL import ImageChops, ImageStat
    except Exception as e:
        sys.exit(f"--verify нужен convert_lib + playwright (как у конвертера): {e}")
    if not img_cols:
        sys.exit("нет Image-колонок для сверки")
    hcols = html_columns(table)
    hcol = "target_html" if "target_html" in hcols else (hcols[0] if hcols else None)
    if hcol is None:
        sys.exit("нет непустой html-колонки (target_html) для рендера")
    icol = img_cols[0]
    os.makedirs(out_dir, exist_ok=True)
    limit = min(n, table.num_rows)
    n_ok = 0
    try:
        for i in range(limit):
            imgs = _row_images(table.column(icol)[i].as_py())
            raw = imgs[0].get("bytes") if imgs else None
            html = table.column(hcol)[i].as_py() or ""
            if not raw or not html:
                print(f"[{i}] пропуск (нет картинки или html)")
                continue
            stored = Image.open(io.BytesIO(raw)).convert("RGB")
            rer = C.render_threaded(html)          # тот же render_full, но через поток (без greenlet-конфликта)
            same_size = stored.size == rer.size
            if same_size:
                mean = sum(ImageStat.Stat(ImageChops.difference(stored, rer)).mean) / 3
                ok = mean < 8                      # эвристика: <8/255 = та же раскладка (мелкий AA-шум шрифтов)
                note = f"сред.разница {mean:.1f}/255"
            else:
                ok = False
                note = "размеры разные"
            n_ok += ok
            print(f"[{i}] {'OK' if ok else 'РАСХОЖДЕНИЕ'}: датасет {stored.size} vs ре-рендер {rer.size} | {note}")
            H = max(stored.height, rer.height)
            canv = Image.new("RGB", (stored.width + rer.width + 20, H), (240, 240, 240))
            canv.paste(stored, (0, 0))
            canv.paste(rer, (stored.width + 20, 0))
            canv.save(os.path.join(out_dir, f"verify_{i:04d}_{'ok' if ok else 'DIFF'}.png"))
    finally:
        C.close_renderer()
    print(f"[verify] совпало {n_ok}/{limit} · side-by-side -> {out_dir}/  "
          f"(размеры разные / большая разница = рендер недетерминирован или target_html не self-contained)")


def main():
    ap = argparse.ArgumentParser(
        description="Просмотр .arrow файлов с поддержкой Image-колонок.")
    ap.add_argument("path", help="путь к .arrow файлу")
    ap.add_argument("-n", type=int, default=5,
                    help="сколько строк показать (по умолчанию 5)")
    ap.add_argument("-c", "--columns", nargs="+",
                    help="показать только эти колонки")
    ap.add_argument("--extract-images", metavar="DIR",
                    help="извлечь все изображения в указанную папку")
    ap.add_argument("--json", action="store_true",
                    help="вывести строки в JSON (картинки как описание)")
    ap.add_argument("--schema-only", action="store_true",
                    help="показать только схему")
    ap.add_argument("--html", metavar="FILE",
                    help="собрать HTML-отчёт: скриншот + код html-колонок")
    ap.add_argument("--verify", type=int, metavar="N",
                    help="сверить рендер: ре-рендерить target_html первых N строк тем же "
                         "рендерером и сравнить со скриншотом (нужны playwright + pytailwindcss)")
    ap.add_argument("--verify-out", metavar="DIR", default="verify",
                    help="куда класть side-by-side сверки (по умолчанию ./verify)")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        sys.exit(f"Файл не найден: {args.path}")

    table = open_arrow(args.path)
    img_cols = image_columns(table)

    if args.extract_images:
        if not img_cols:
            print("В файле нет Image-колонок для извлечения.")
        else:
            extract_images(table, img_cols, args.extract_images)
        return

    if args.verify is not None:
        verify_render(table, img_cols, args.verify_out, args.verify)
        return

    if args.html:
        h_cols = html_columns(table)
        if args.columns:  # уважаем -c, если задан
            h_cols = [c for c in h_cols if c in args.columns]
        build_html_report(table, args.html, img_cols, h_cols, args.n)
        return

    print_schema(table, img_cols)
    if args.schema_only:
        return

    if args.json:
        dump_json(table, args.n, args.columns, img_cols)
    else:
        print_rows(table, args.n, args.columns, img_cols)


if __name__ == "__main__":
    main()