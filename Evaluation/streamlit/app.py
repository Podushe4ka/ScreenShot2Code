"""
app.py — Streamlit-демо ScreenShot2Code.

Пользователь загружает скриншот веб-страницы, дообученный чекпоинт
Qwen3.5-4B (обычный transformers, без vLLM — для единичных запросов
батчевый сервер не нужен) генерирует HTML/Tailwind код, который затем
рендерится обратно в скриншот (тем же движком — render.py из основного
пайплайна) и показывается рядом с оригиналом.

Запуск (НА СЕРВЕРЕ С GPU):
    streamlit run app.py -- --model /mnt/storage-1/ScreenShot2Code/model_weights/<run>/<step>

Доступ со своего компьютера — через SSH-туннель (модель и Playwright
остаются на сервере, тоннель только пробрасывает порт):
    ssh -L 8501:localhost:8501 user@gpu-server
    # затем открыть http://localhost:8501 в локальном браузере

См. README.md рядом с этим файлом для деталей.
"""

import argparse
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# render.py лежит рядом с этим файлом (копия/симлинк из основного пайплайна
# ScreenShot2Code) — переиспользуем ТОТ ЖЕ рендерер, что и в бенчмарке, чтобы
# скриншоты предсказания были получены идентичным образом (тот же Chromium,
# те же флаги, та же замена <img> на плейсхолдер).
from render import prepare_and_render

# =============================================================================
# Промпт — 1-в-1 из run_benchmark_batched.py, не менять без причины: модель
# дообучалась/оценивалась именно на этом тексте, другой промпт даст другое
# распределение выходов.
# =============================================================================
PROMPT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, "
    "no network requests) that reproduces the layout, text, and colors as closely as "
    "possible. Use plain gray placeholder boxes instead of any real images. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)


def parse_args():
    """Streamlit прокидывает всё после `--` в sys.argv как есть — здесь
    обычный argparse, ничего специфичного для Streamlit не нужно."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="HF-путь или локальный путь к чекпоинту (например, "
             "/mnt/storage-1/ScreenShot2Code/model_weights/<run>/<step>/).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--device", default="cuda",
        help="cuda / cuda:0 / cpu. Внутри Docker-контейнера, запущенного через "
             "run.sh с GPUS='\"device=N\"', видна только ОДНА GPU, и она всегда "
             "cuda:0 — 'cuda' здесь эквивалентен 'cuda:0', менять не нужно.",
    )
    # Streamlit сам съедает свои аргументы до "--", так что здесь мы видим
    # только то, что пользователь передал после него.
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args


ARGS = parse_args()


# =============================================================================
# Загрузка модели — один раз на весь процесс сервера, кэшируется Streamlit'ом
# между запросами разных пользователей/перезапусков скрипта (реран при любом
# клике в UI НЕ должен перегружать 4B чекпоинт с диска каждый раз).
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()
    return model, processor


def generate_html(model, processor, image: Image.Image, device: str, max_new_tokens: int) -> tuple[str, float]:
    """Прогоняет одно изображение через модель, возвращает (extracted_html, seconds)."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    chat_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_prompt], images=[image], return_tensors="pt").to(device)

    t0 = time.monotonic()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.monotonic() - t0

    # Отрезаем входной промпт из вывода — generate() возвращает prompt+completion вместе.
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return extract_html(raw_text), elapsed


def extract_html(text: str) -> str:
    """1-в-1 из run_benchmark_batched.py — снимает <think>, markdown-фенсы,
    обрезает всё до <!doctype/<html, если модель добавила преамбулу."""
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    low = text.lower()
    start = low.find("<!doctype")
    if start == -1:
        start = low.find("<html")
    if start != -1:
        text = text[start:]
    return text.strip()


# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="ScreenShot2Code — демо", layout="wide")

st.title("🖼️ → 🧑‍💻 ScreenShot2Code")
st.caption(
    f"Модель: `{ARGS.model}` · устройство: `{ARGS.device}` · "
    "загрузите скриншот веб-страницы — модель сгенерирует HTML/Tailwind код, "
    "который затем рендерится обратно в картинку для сравнения."
)

with st.spinner(f"Загружаю модель {ARGS.model} (один раз при старте сервера)…"):
    model, processor = load_model(ARGS.model, ARGS.device)

uploaded = st.file_uploader(
    "Скриншот веб-страницы (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"]
)

col_settings1, col_settings2 = st.columns(2)
with col_settings1:
    max_new_tokens = st.slider("Максимум токенов генерации", 512, 8192, ARGS.max_new_tokens, step=256)
with col_settings2:
    run_button = st.button("Сгенерировать код ▶", type="primary", disabled=uploaded is None)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    if run_button:
        work_dir = Path("streamlit_runs") / str(int(time.time()))
        work_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Модель генерирует HTML…"):
            html_code, gen_seconds = generate_html(
                model, processor, image, ARGS.device, max_new_tokens
            )

        with st.spinner("Рендерю сгенерированный HTML обратно в скриншот…"):
            html_path = work_dir / "pred.html"
            png_path = work_dir / "pred.png"
            render_info = prepare_and_render(html_code, str(html_path), str(png_path))

        st.success(f"Готово за {gen_seconds:.1f} с.")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Оригинал")
            st.image(image, use_container_width=True)
        with col_right:
            st.subheader("Рендер сгенерированного кода")
            if render_info["render_ok"]:
                st.image(str(png_path), use_container_width=True)
            else:
                st.error("Рендер не удался — см. код ниже, возможно синтаксическая ошибка в HTML.")

        if render_info["n_images_replaced"] > 0:
            st.caption(
                f"ℹ️ {render_info['n_images_replaced']} тег(ов) <img> заменены на серые "
                "плейсхолдеры (как в основном пайплайне метрик)."
            )

        with st.expander("Сгенерированный HTML-код", expanded=False):
            st.code(html_code, language="html")
            st.download_button(
                "Скачать HTML",
                data=html_code,
                file_name="generated.html",
                mime="text/html",
            )
    else:
        st.subheader("Оригинал")
        st.image(image, use_container_width=True)
        st.info("Нажмите «Сгенерировать код ▶», чтобы прогнать модель.")
else:
    st.info("⬆️ Загрузите скриншот, чтобы начать.")
