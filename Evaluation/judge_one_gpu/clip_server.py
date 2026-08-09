"""
clip_server.py — постоянный процесс с ОДНОЙ копией CLIP-модели на GPU,
батчующий inference-запросы от N параллельных CPU-воркеров рендера/метрик.

Проблема, которую это решает: если CLIP грузится напрямую в каждом воркере
--num-workers процессов (ProcessPoolExecutor) — это до num_workers копий
весов в VRAM одновременно, и каждый forward идёт с batch_size=1, GPU большую
часть времени просто ждёт по одному изображению за раз от каждого воркера —
классический anti-pattern для GPU-инференса (throughput определяется
размером батча, а не числом процессов, которые его дёргают).

Архитектура:
  - ОДИН процесс ClipServer, запускается один раз в начале run_benchmark_batched
    (после загрузки vLLM) и живёт до конца всего прогона — не пересоздаётся
    на каждый батч из --batch-size сэмплов.
  - CPU-воркеры (render_and_score_one, в своих процессах ProcessPoolExecutor)
    не грузят CLIP вообще. Вместо локального инференса они кладут запрос
    (две картинки + два списка bbox-блоков) в общую request_queue и
    блокирующе ждут ответ на СВОЁМ приватном Pipe — с точки зрения кода
    score_pair() это выглядит как обычный синхронный вызов, вся батчевая
    механика спрятана в ClipClient.
  - ClipServer копит запросы из request_queue до --clip-batch-size ИЛИ до
    истечения короткого таймаута (см. _BATCH_TIMEOUT_SEC) — так неполный
    батч в конце потока запросов не зависает в ожидании остальных
    clip_batch_size-1 запросов, которые могут не прийти ещё секунды (это
    бывает под конец батча из --batch-size сэмплов, когда CPU-воркеры уже
    почти все закончили и новые CLIP-запросы прилетают редко).
  - Один forward encode_image() на весь собранный под-батч, результаты
    раздаются обратно по приватным Pipe'ам запросивших воркеров.

Использование:
    # в главном процессе run_benchmark_batched.py, один раз:
    from clip_server import ClipServer
    clip_server = ClipServer(clip_batch_size=256)
    clip_server.start()
    ...
    # передать clip_server.request_queue в воркеры через initializer пула
    ...
    clip_server.stop()

    # в воркере (render_and_score_one), один раз на процесс, через initializer:
    from clip_server import ClipClient
    _clip_client = ClipClient(request_queue)
    # затем в metrics.py:
    metrics.set_clip_client(_clip_client)
"""

import multiprocessing as mp
import queue
import time
import uuid

# Таймаут ожидания неполного под-батча на сервере: если запросов меньше
# clip_batch_size, не ждём остальные бесконечно — набираем что есть за это
# время и считаем. Короткий, чтобы не раздувать latency отдельных сэмплов
# (типичный случай — "хвост" батча, когда CPU-воркеры уже почти все
# закончили и новые запросы приходят редко).
_BATCH_TIMEOUT_SEC = 0.075


class ClipRequest:
    __slots__ = ("request_id", "image_path1", "image_path2", "bboxes1", "bboxes2", "reply_conn")

    def __init__(self, request_id, image_path1, image_path2, bboxes1, bboxes2, reply_conn):
        self.request_id = request_id
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.bboxes1 = bboxes1
        self.bboxes2 = bboxes2
        self.reply_conn = reply_conn


def _server_loop(request_queue: mp.Queue, clip_batch_size: int, ready_event, device_str: str,
                  gpu_index: str = None):
    """Тело процесса-сервера. Импорт torch/clip только здесь — не в главном
    процессе (тот же принцип, что и в остальном коде: тяжёлые GPU-импорты
    внутри функции, которая реально выполняется в целевом процессе).

    gpu_index (если задан) проставляет CUDA_VISIBLE_DEVICES ДО импорта torch
    в этом процессе — CLIP и vLLM делят одну физическую GPU (см. RUNNING.md),
    без явного pin сюда могла бы попасть GPU 0 хоста по умолчанию torch, даже
    если --gpu-index указывает на другую карту, на которой реально крутится
    vllm serve."""
    if gpu_index is not None:
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    import torch
    import clip as clip_pkg
    from PIL import Image

    # _rescale_and_mask копируем сюда же (не импортируем metrics.py, чтобы
    # не тащить весь его импорт-тайм груз — cv2/colormath/torch-clip второй
    # раз — в этот процесс; логика идентична _rescale_and_mask в metrics.py).
    def _mask_bounding_boxes_with_inpainting(image, bounding_boxes):
        import cv2
        import numpy as np
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

    def _rescale_and_mask(image_path, bboxes):
        with Image.open(image_path) as img:
            if len(bboxes) > 0:
                img = _mask_bounding_boxes_with_inpainting(img, bboxes)
            width, height = img.size
            new_size = (width, width) if width < height else (height, height)
            return img.resize(new_size, Image.LANCZOS)

    device = device_str
    print(f"[clip_server] Загружаю CLIP ViT-B/32 на {device} (один раз на весь прогон)...")
    model, preprocess = clip_pkg.load("ViT-B/32", device=device)
    model.eval()
    ready_event.set()
    print(f"[clip_server] Готов, clip_batch_size={clip_batch_size}, "
          f"batch_timeout={_BATCH_TIMEOUT_SEC}s")

    while True:
        # Блокирующе ждём первый запрос под-батча — если это команда
        # остановки (None), выходим.
        first = request_queue.get()
        if first is None:
            print("[clip_server] Получена команда остановки, завершаюсь.")
            return

        batch = [first]
        deadline = time.monotonic() + _BATCH_TIMEOUT_SEC
        while len(batch) < clip_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                # Команда остановки пришла в середине набора под-батча —
                # досчитываем уже собранный батч, затем выходим на
                # следующей итерации внешнего while.
                request_queue.put(None)
                break
            batch.append(item)

        # --- препроцессинг всего под-батча ---
        tensors1, tensors2, valid = [], [], []
        for req in batch:
            try:
                img1 = preprocess(_rescale_and_mask(req.image_path1, req.bboxes1))
                img2 = preprocess(_rescale_and_mask(req.image_path2, req.bboxes2))
                tensors1.append(img1)
                tensors2.append(img2)
                valid.append(req)
            except Exception as e:
                # Битый рендер/файл — не роняем весь под-батч, просто
                # отвечаем этому конкретному запросу ошибкой (score_pair на
                # стороне воркера словит это как metric_error).
                try:
                    req.reply_conn.send(("error", str(e)))
                    req.reply_conn.close()
                except Exception:
                    pass

        if not valid:
            continue

        # --- один batched forward на весь под-батч ---
        try:
            batch1 = torch.stack(tensors1).to(device)
            batch2 = torch.stack(tensors2).to(device)
            with torch.no_grad():
                f1 = model.encode_image(batch1)
                f2 = model.encode_image(batch2)
            f1 = f1 / f1.norm(dim=-1, keepdim=True)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            sims = (f1 * f2).sum(dim=-1).tolist()  # поэлементный cosine sim, эквивалент (f1 @ f2.T).diag()
        except Exception as e:
            for req in valid:
                try:
                    req.reply_conn.send(("error", str(e)))
                    req.reply_conn.close()
                except Exception:
                    pass
            continue

        for req, sim in zip(valid, sims):
            try:
                req.reply_conn.send(("ok", float(sim)))
                req.reply_conn.close()
            except Exception:
                pass  # воркер уже мог упасть/отсоединиться — не критично


class ClipServer:
    """Хендл главного процесса: старт/стоп процесса-сервера + доступ к
    request_queue, который нужно передать воркерам (через initializer пула
    процессов — см. run_benchmark_batched.py)."""

    def __init__(self, clip_batch_size: int = 256, device: str = None, gpu_index: str = None):
        """gpu_index: индекс физической GPU (для CUDA_VISIBLE_DEVICES в
        дочернем процессе) ДОЛЖЕН совпадать с --gpu-index, который
        используется для vllm serve (см. run_benchmark_batched.py/
        vllm_server_manager.py), иначе CLIP и vLLM окажутся на разных
        физических картах хоста. Если None, процесс наследует
        CUDA_VISIBLE_DEVICES от родителя."""
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_batch_size = clip_batch_size
        self.device = device
        self.gpu_index = gpu_index
        ctx = mp.get_context("spawn")
        self.request_queue: mp.Queue = ctx.Queue()
        self._ready_event = ctx.Event()
        self._process = ctx.Process(
            target=_server_loop,
            args=(self.request_queue, self.clip_batch_size, self._ready_event, self.device, self.gpu_index),
            daemon=True,
        )

    def start(self, wait_ready: bool = True, timeout: float = 120.0):
        self._process.start()
        if wait_ready:
            if not self._ready_event.wait(timeout=timeout):
                raise RuntimeError(
                    f"[clip_server] Не поднялся за {timeout}с — проверьте, что "
                    f"свободна VRAM и модель ViT-B/32 доступна (HF_HOME/кэш)."
                )

    def stop(self, timeout: float = 30.0):
        if self._process.is_alive():
            self.request_queue.put(None)
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                print("[clip_server] Не завершился штатно за таймаут, убиваю принудительно.")
                self._process.terminate()


class ClipClient:
    """Живёт внутри воркера (один экземпляр на процесс, создаётся в
    initializer пула). Синхронный интерфейс для score_pair: сам факт
    батчевания на сервере полностью скрыт за этим блокирующим вызовом."""

    def __init__(self, request_queue: mp.Queue):
        self.request_queue = request_queue

    def similarity(self, image_path1: str, image_path2: str, bboxes1: list, bboxes2: list) -> float:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        req = ClipRequest(
            request_id=uuid.uuid4().hex,
            image_path1=image_path1, image_path2=image_path2,
            bboxes1=bboxes1, bboxes2=bboxes2,
            reply_conn=child_conn,
        )
        self.request_queue.put(req)
        status, payload = parent_conn.recv()
        parent_conn.close()
        if status == "error":
            raise RuntimeError(f"[clip_client] Ошибка CLIP-сервера: {payload}")
        return payload
