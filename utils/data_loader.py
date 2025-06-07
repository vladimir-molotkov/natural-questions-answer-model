import os
from pathlib import Path

import dvc.api
from datasets import DatasetDict, load_dataset


def get_dvc_params():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"
    return dvc.api.params_show(str(config_path))


def download_nq_data():
    """Download and cache Natural Questions dataset"""
    data_dir = "data/natural_questions"
    os.makedirs(data_dir, exist_ok=True)

    # Load parameters from params.yaml
    params = get_dvc_params()
    sample_size = params["data"]["sample_size"]

    print(f"Downloading Natural Questions dataset (sample: {sample_size})...")

    train_data = load_dataset(
        "google-research-datasets/natural_questions", split="train"
    )
    val_data = load_dataset(
        "google-research-datasets/natural_questions", split="validation"
    )

    train_data = train_data.map(preprocess_nq, remove_columns=train_data.column_names)
    val_data = val_data.map(preprocess_nq, remove_columns=val_data.column_names)

    train_data = train_data.filter(lambda x: x["answer"] != "")
    val_data = val_data.filter(lambda x: x["answer"] != "")

    if sample_size:
        train_data = train_data.select(range(min(sample_size, len(train_data))))
        val_data = val_data.select(range(min(sample_size, len(val_data))))

    # Save to local
    dataset = DatasetDict({"train": train_data, "validation": val_data})

    dataset.save_to_disk(data_dir)
    print(f"Dataset saved to {data_dir}")


# def load_local_nq_data(split="train"):
#     """Load dataset from local storage"""
#     data_dir = "data/natural_questions"
#     if not os.path.exists(data_dir):
#         download_nq_data()

#     dataset = DatasetDict.load_from_disk(data_dir)
#     return dataset[split]


def preprocess_nq(example):
    """Extract first short answer from Natural Questions examples with complete error handling"""

    result = {"id": example.get("id", ""), "question": "", "context": "", "answer": ""}

    if "question" in example and isinstance(example["question"], dict):
        result["question"] = example["question"].get("text", "")

    if "document" in example and "tokens" in example["document"]:
        result["context"] = " ".join(example["document"]["tokens"])

    if (
        "annotations" in example
        and isinstance(example["annotations"], list)
        and len(example["annotations"]) > 0
    ):
        first_annotation = example["annotations"][0]
        if (
            isinstance(first_annotation, dict)
            and "short_answers" in first_annotation
            and isinstance(first_annotation["short_answers"], list)
            and len(first_annotation["short_answers"]) > 0
            and isinstance(first_annotation["short_answers"][0], dict)
        ):
            result["answer"] = first_annotation["short_answers"][0].get("text", "")

    return result


def load_local_nq_data(split="train"):
    """Load dataset from local storage with validation"""
    data_dir = Path("data/natural_questions")
    try:
        dataset = DatasetDict.load_from_disk(str(data_dir))
        if split not in dataset:
            raise ValueError(f"Split {split} not found in local dataset")
        return dataset[split]
    except Exception as e:
        print(f"Local data loading failed: {str(e)}")
        raise


def get_nq_data(split="train", sample_size=None):
    """Main data loading function with automatic fallback"""
    try:
        return load_local_nq_data(split)
    except Exception:
        print("Downloading fresh dataset...")
        download_nq_data()
        return load_local_nq_data(split)
