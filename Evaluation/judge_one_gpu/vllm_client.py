"""
vllm_client.py — тонкий HTTP-клиент к vLLM OpenAI-совместимому серверу.

Один и тот же base_url переиспользуется последовательно для checkpoint,
baseline и judge (см. vllm_server_manager.py) — какая модель сейчас отвечает
на порту, решается ДО того как этот клиент используется, сам клиент об этом
не знает и не должен.

Кодирование изображений: vLLM chat API принимает image_url с data: URI
(base64), как OpenAI API. PIL.Image -> PNG-байты -> base64 делается в
воркер-потоке пула, непосредственно перед отправкой конкретного запроса
(не заранее одним последовательным списком на весь батч до первого запроса)
— иначе для датасетов с крупными full-page скриншотами (WebSight) кодирование
тысяч картинок последовательно перед первым HTTP-запросом визуально
неотличимо от зависания: GPU 0%, логи vLLM и stdout молчат, потому что до
сервера в этот момент ещё ничего не дошло. При max_concurrency потоках
кодирование идёт параллельно с самими запросами, и первый запрос уходит
почти сразу.

Сетевые сбои (timeout, обрыв соединения) ретраятся с конечным числом попыток
— вместо того чтобы терять сэмпл насовсем при однократном таймауте. Ошибки с
содержательным телом ответа (4xx/5xx от самого vLLM — например, невалидный
запрос) не ретраятся: это не транзиент, а вероятная реальная проблема с
запросом или моделью, повтор её не исправит.
"""

import base64
import io
import time
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm


class VLLMServerError(RuntimeError):
    pass


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _image_path_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


class VLLMClient:
    """Клиент к одному vLLM OpenAI-совместимому серверу на base_url."""

    def __init__(self, base_url: str, model_name: str, timeout: float = 300.0,
                 max_pool_size: int = 128, max_retries: int = 3,
                 retry_backoff_sec: float = 5.0):
        """max_pool_size: requests.Session по умолчанию держит пул всего на
        10 соединений, а этот клиент открывает до --generation-concurrency
        (по умолчанию 64) или --num-workers (может быть больше сотни)
        конкурентных запросов через один Session — с дефолтным пулом
        соединения выше лимита не переиспользуются эффективно. Ставим пул
        явно с запасом над типичной конкурентностью.

        max_retries: сколько ДОПОЛНИТЕЛЬНЫХ попыток делать при таймауте или
        обрыве соединения (то есть всего до max_retries + 1 попыток). Не
        бесконечно — при системной проблеме (например, сервер реально упал —
        EngineDeadError) конечный потолок не даёт одному сэмплу держать пул
        вечно, просто быстрее исчерпывает попытки и возвращает ошибку как
        раньше. retry_backoff_sec: пауза между попытками, растёт линейно с
        номером попытки (backoff), чтобы не долбить сервер, если проблема
        временная перегрузка, а не единичный сетевой сбой."""
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=max_pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def wait_until_ready(self, max_wait_s: float = 900.0, poll_interval_s: float = 5.0) -> None:
        """Блокирующе ждёт, пока сервер не ответит на /health."""
        deadline = time.monotonic() + max_wait_s
        last_err = None
        while time.monotonic() < deadline:
            try:
                resp = self._session.get(f"{self.base_url}/health", timeout=10.0)
                if resp.status_code == 200:
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll_interval_s)
        raise VLLMServerError(
            f"Сервер {self.base_url} ({self.model_name}) не поднялся за {max_wait_s}с. "
            f"Последняя ошибка: {last_err}"
        )

    def chat(
        self,
        messages: list,
        max_tokens: int,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
    ) -> dict:
        """Один запрос /v1/chat/completions, с retry на транзиентные сетевые
        сбои (Timeout/ConnectionError). HTTP-ответ с ненулевым статусом от
        самого сервера (запрос дошёл, но vLLM его отверг) НЕ ретраится —
        поднимается сразу как VLLMServerError."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)

        last_exc = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self._session.post(
                    f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt <= self.max_retries:
                    print(f"[vllm_client] {self.model_name}: попытка {attempt} упала "
                          f"({e.__class__.__name__}), повтор через "
                          f"{self.retry_backoff_sec * attempt:.0f}с...")
                    time.sleep(self.retry_backoff_sec * attempt)
                    continue
                raise VLLMServerError(
                    f"[{self.model_name}] Сетевой сбой после {attempt} попыток: {e!r}"
                ) from e

            if resp.status_code != 200:
                raise VLLMServerError(
                    f"[{self.model_name}] HTTP {resp.status_code} от {self.base_url}: {resp.text[:2000]}"
                )
            return resp.json()

        # Не должно достигаться (цикл либо возвращает, либо кидает выше),
        # оставлено как явная защита от молчаливого None.
        raise VLLMServerError(f"[{self.model_name}] Запрос не выполнен: {last_exc!r}")

    def chat_batch(
        self,
        images: list,
        text: str,
        max_tokens: int,
        temperature: float = 0.0,
        max_concurrency: int = 64,
        extra_body: Optional[dict] = None,
        progress_label: Optional[str] = None,
    ) -> list[Optional[str]]:
        """Отправляет много независимых чатов (одна картинка + text на
        каждый) конкурентно через ThreadPoolExecutor — запросы I/O-bound на
        клиенте, тяжёлая работа батчуется внутри vLLM через continuous
        batching на сервере. Кодирование картинки в base64 происходит в
        рабочем потоке, непосредственно перед отправкой (см. docstring
        модуля) — не заранее одним списком.

        Возвращает список текстов ответов в том же порядке, что images;
        None на месте запроса, упавшего даже после retry в chat() (ошибка
        не роняет остальные — вызывающий код трактует None как
        generation_error)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[Optional[str]] = [None] * len(images)

        def _one(i: int, image):
            try:
                messages = [{"role": "user", "content": image_message_content(image, text)}]
                resp = self.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                  extra_body=extra_body)
                results[i] = resp["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[vllm_client] Запрос {i} к {self.model_name} окончательно упал: {e}")

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [pool.submit(_one, i, img) for i, img in enumerate(images)]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=progress_label,
                          disable=not progress_label, mininterval=15.0):
                pass

        return results


def image_message_content(image: Image.Image, text: str) -> list:
    """Собирает content-блок OpenAI chat формата: одна картинка (PIL.Image,
    инлайнится как base64 data URI) + текст."""
    return [
        {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
        {"type": "text", "text": text},
    ]


def image_path_message_part(path: str) -> dict:
    return {"type": "image_url", "image_url": {"url": _image_path_to_data_uri(path)}}
