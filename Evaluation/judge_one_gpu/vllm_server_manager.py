"""
vllm_server_manager.py — управляет ОДНИМ процессом `vllm serve` на ОДНОЙ GPU,
последовательно переключая его между тремя моделями (checkpoint, baseline,
judge) внутри каждого батча.

Физически одна GPU — держать на ней три модели одновременно нельзя, поэтому
на каждом батче цикл:
    1. поднять checkpoint  -> сгенерировать HTML по всему батчу -> погасить
    2. поднять baseline    -> сгенерировать HTML по всему батчу -> погасить
    3. поднять judge       -> прогнать pairwise-сравнение по батчу -> погасить
    (рендер+метрики чекпоинта считаются между шагами 1 и 2, пока GPU
    свободен от vLLM — CLIP-сервер, отдельный маленький GPU-процесс, живёт
    всё это время постоянно, см. clip_server.py и run_benchmark_batched.py)

run_benchmark_batched.py стартует/останавливает `vllm serve` через
subprocess.Popen, ждёт готовности через /health после КАЖДОГО переключения
модели и посылает SIGTERM+SIGKILL-fallback перед стартом следующей.

ВАЖНО про VRAM: CLIP (~350MB) держится на GPU постоянно, поэтому у vLLM
здесь --gpu-memory-utilization должен оставлять запас под него (см.
DEFAULT_GPU_MEMORY_UTILIZATION, переопределяется через
VLLM_GPU_MEMORY_UTILIZATION или run.sh).

Между остановкой одного vLLM-процесса и стартом следующего — kill + короткая
пауза (см. _SHUTDOWN_GRACE_SEC), без явного опроса nvidia-smi на
освобождение VRAM. Если освобождение занимает больше паузы на какой-то
среде — первый симптом: следующий vllm serve падает с OOM при загрузке
весов, тогда стоит увеличить _SHUTDOWN_GRACE_SEC.
"""

import os
import signal
import subprocess
import sys
import time

import requests

DEFAULT_GPU_MEMORY_UTILIZATION = 0.85

# Пауза после kill предыдущего vLLM-процесса перед стартом следующего — даёт
# CUDA-контексту и NCCL реально освободить VRAM. Без явной проверки
# nvidia-smi (см. docstring модуля) — просто фиксированная задержка.
_SHUTDOWN_GRACE_SEC = float(os.environ.get("D2C_VLLM_SHUTDOWN_GRACE_SEC", "10"))

# Сколько ждём SIGTERM, прежде чем добить SIGKILL — vLLM на SIGTERM обычно
# завершается за пару секунд (нет чекпоинтов/состояния, которые нужно
# сбрасывать на диск), но даём запас на выгрузку CUDA-контекста.
_SIGTERM_TIMEOUT_SEC = 30.0


class VLLMServerError(RuntimeError):
    pass


class VLLMServerManager:
    """Держит на ОДНОЙ GPU ровно один `vllm serve` процесс за раз. switch_to()
    гасит текущую модель (если есть) и поднимает новую, блокируясь до её
    готовности (/health). Один и тот же base_url/port переиспользуется для
    всех трёх моделей — с точки зрения VLLMClient (vllm_client.py) это
    просто "сервер стал отвечать другой моделью", сам HTTP-клиент не знает
    о переключениях."""

    def __init__(self, port: int = 8001, gpu_index: str = "0",
                 gpu_memory_utilization: float = None, max_model_len: int = 16384,
                 log_dir: str = None, extra_served_args: list = None):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.gpu_index = gpu_index
        self.gpu_memory_utilization = (
            gpu_memory_utilization if gpu_memory_utilization is not None
            else float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", DEFAULT_GPU_MEMORY_UTILIZATION))
        )
        self.max_model_len = max_model_len
        self.log_dir = log_dir
        self.extra_served_args = extra_served_args or []

        self._proc: subprocess.Popen = None
        self._current_model: str = None
        self._current_log_file = None

    @property
    def current_model(self) -> str:
        return self._current_model

    def switch_to(self, model_name: str, label: str, log_level: str = "INFO",
                   ready_timeout_s: float = 1200.0) -> None:
        """Гасит текущую модель (если отличается или уже что-то поднято) и
        поднимает model_name, блокируясь до готовности. Если model_name уже
        текущая (не должно происходить в нашем цикле checkpoint->baseline->
        judge->checkpoint..., но защищаемся на случай будущих изменений
        порядка этапов) — просто не делает ничего, чтобы не тратить время на
        бесполезный перезапуск."""
        if self._current_model == model_name and self._proc is not None \
                and self._proc.poll() is None:
            return

        self.stop()

        print(f"[vllm_server_manager] Поднимаю '{label}' ({model_name}) на "
              f"GPU {self.gpu_index}, порт {self.port}, "
              f"gpu-memory-utilization={self.gpu_memory_utilization} ...")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_index
        env["VLLM_LOGGING_LEVEL"] = log_level

        cmd = [
            "vllm", "serve", model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--limit-mm-per-prompt", '{"image": 3}',
            *self.extra_served_args,
        ]

        log_file = None
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"{label}.log")
            log_file = open(log_path, "a", buffering=1, encoding="utf-8")
            log_file.write(f"\n\n=== [{label}] запуск {model_name} "
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            stdout_target = log_file
            stderr_target = log_file

        proc = subprocess.Popen(cmd, env=env, stdout=stdout_target, stderr=stderr_target)

        self._proc = proc
        self._current_model = model_name
        self._current_log_file = log_file

        self._wait_ready(label, model_name, ready_timeout_s)

    def _wait_ready(self, label: str, model_name: str, max_wait_s: float,
                     early_fail_s: float = 5.0, poll_interval_s: float = 5.0) -> None:
        """Ждёт /health, но сначала быстро проверяет, что процесс не упал
        сразу (битый путь к модели, немедленный OOM при загрузке) — так
        понятная ошибка приходит за секунды, а не после полного
        max_wait_s ожидания /health, который никогда не ответит."""
        time.sleep(early_fail_s)
        ret = self._proc.poll()
        if ret is not None:
            self._raise_dead(label, model_name, ret)

        deadline = time.monotonic() + max_wait_s
        last_err = None
        while time.monotonic() < deadline:
            ret = self._proc.poll()
            if ret is not None:
                self._raise_dead(label, model_name, ret)
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=10.0)
                if resp.status_code == 200:
                    print(f"[vllm_server_manager] '{label}' готов ({model_name}).")
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll_interval_s)

        self.stop()
        raise VLLMServerError(
            f"[{label}] Сервер ({model_name}) не поднялся за {max_wait_s}с. "
            f"Последняя ошибка: {last_err}"
        )

    def _raise_dead(self, label: str, model_name: str, returncode: int) -> None:
        tail = ""
        if self._current_log_file is not None:
            try:
                self._current_log_file.flush()
                with open(self._current_log_file.name, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                tail = "".join(lines[-40:])
            except Exception:
                pass
        self._cleanup_handles()
        raise VLLMServerError(
            f"[{label}] Процесс vllm serve ({model_name}) завершился сразу "
            f"(returncode={returncode}), не дождавшись готовности.\n"
            f"Последние строки лога:\n{tail}"
        )

    def stop(self, timeout: float = _SIGTERM_TIMEOUT_SEC) -> None:
        """Гасит текущий vLLM-процесс (если есть) и ждёт _SHUTDOWN_GRACE_SEC
        перед возвратом — даёт VRAM освободиться до того, как вызывающий код
        (switch_to) попытается поднять следующую модель на той же GPU."""
        if self._proc is not None:
            if self._proc.poll() is None:
                print(f"[vllm_server_manager] Останавливаю '{self._current_model}' "
                      f"(pid {self._proc.pid})...")
                try:
                    self._proc.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(f"[vllm_server_manager] '{self._current_model}' не завершился "
                          f"за {timeout}с, добиваю SIGKILL.")
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=15.0)
                    except Exception:
                        pass
            self._cleanup_handles()
            time.sleep(_SHUTDOWN_GRACE_SEC)

    def _cleanup_handles(self) -> None:
        if self._current_log_file is not None:
            try:
                self._current_log_file.close()
            except Exception:
                pass
        self._proc = None
        self._current_model = None
        self._current_log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
