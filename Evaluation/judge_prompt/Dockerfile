# Dockerfile — окружение только для judge-экспериментов: vLLM (сервер +
# клиент) и Pillow/requests. Никакого рендеринга (playwright/chromium) и
# никакого CLIP — та часть эксперимента (сравнение картинок с эталоном) уже
# сделана заранее и лежит на диске как pred_baseline.png/pred_checkpoint.png,
# сюда эти шаги не нужны.
FROM vllm/vllm-openai:latest

WORKDIR /app

# requests уже тянется транзитивно вместе с vllm, но фиксируем явно —
# дешевле одна лишняя строка в pip install, чем ловить ImportError через
# полгода, если vllm перестанет тянуть requests как зависимость.
RUN pip install --no-cache-dir requests pillow

COPY vllm_client.py judge_client.py sample_order.py prompts.py run_judge_eval.py split_data.py ./
COPY serve_judge.sh ./
RUN chmod +x serve_judge.sh

ENTRYPOINT ["/app/serve_judge.sh"]
