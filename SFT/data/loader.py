from datasets import load_from_disk, Dataset

def load_sft_dataset(path: str) -> Dataset:
    dataset = load_from_disk(path)
    dataset_filtered = dataset.filter(lambda ex: ex['task_type'] == 'drafting')
    if len(dataset_filtered) == 0:
        raise ValueError(f"No drafting samples in {path}")
    return dataset_filtered
# когда будет готово -- добавить считывание всех видов task_type