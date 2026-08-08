"""
vllm_client.py — тонкий HTTP-клиент к ОДНОМУ vLLM OpenAI-совместимому
серверу (судья). Здесь, в отличие от основного eval-пайплайна, нужен только
один сервер за раз — во время подбора промпта ты гоняешь одну judge-модель
(2B/4B/9B/27B, см. serve_judge.sh) и меняешь промпт между запусками
run_judge_eval.py, не поднимая ничего дополнительного.

Изображения кодируются как data: URI (base64), как в OpenAI API — так же,
как ожидает vLLM chat API.
"""

import base64
import time
from pathlib import Path
from typing import Optional

import requests


class VLLMServerError(RuntimeError):
    pass


def _image_path_to_data_uri(path: Path) -> str:
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_path_message_part(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": _image_path_to_data_uri(path)}}


class VLLMClient:
    def __init__(self, base_url: str, model_name: str, timeout: float = 300.0,
                 max_pool_size: int = 64):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=max_pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def wait_until_ready(self, max_wait_s: float = 900.0, poll_interval_s: float = 5.0) -> None:
        """Блокирующе ждёт /health — vLLM отвечает 200 только после полной
        загрузки весов, так что первый chat-запрос не должен уходить раньше."""
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

    def chat(self, messages: list, max_tokens: int, temperature: float = 0.0,
              extra_body: Optional[dict] = None) -> dict:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)
        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise VLLMServerError(
                f"[{self.model_name}] HTTP {resp.status_code} от {self.base_url}: {resp.text[:2000]}"
            )
        return resp.json()
