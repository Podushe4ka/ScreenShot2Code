DRAFTING_PROMPT = "Generate a single self-contained HTML file with precompiled Tailwind. Replace images with gray placeholder blocks."

MIN_PIXELS = 256 * 32 * 32
MAX_PIXELS = 1280 * 32 * 32

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
THINK_END = "</think>"


# добавить другие виды обучающих примеров
def to_message(example):
    if example["task_type"] == "drafting":
        html_gt = example["target_html"]
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": DRAFTING_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": html_gt}],
            },
        ]
    return example


def _find_subsequence(seq: list[int], sub: list[int]) -> int | None:
    if not sub:
        return None
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


def make_collate_fn(processor):
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    response_token_ids = processor.tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    think_end_ids = processor.tokenizer.encode(THINK_END, add_special_tokens=False)
    pad_token_id = processor.tokenizer.pad_token_id

    def collate_fn(examples: list) -> dict:
        texts = [
            processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            for ex in examples
        ]
        images = [ex["images"] for ex in examples]
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        labels[labels == image_token_id] = -100

        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()
            start = _find_subsequence(ids, response_token_ids)
            if start is None:
                labels[i, :] = -100
                continue
            boundary = start + len(response_token_ids)
            end_of_think = _find_subsequence(ids[boundary:], think_end_ids)
            if end_of_think is not None:
                boundary += end_of_think + len(think_end_ids)
                while (
                    boundary < len(ids)
                    and not processor.tokenizer.decode([ids[boundary]]).strip()
                ):
                    boundary += 1
            labels[i, :boundary] = -100
        batch["labels"] = labels
        return batch

    return collate_fn
