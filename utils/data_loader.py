import os

import dvc.api
from datasets import DatasetDict, load_dataset


def download_nq_data():
    """Download and cache Natural Questions dataset"""
    data_dir = "data/natural_questions"
    os.makedirs(data_dir, exist_ok=True)

    # Load parameters from params.yaml
    params = dvc.api.params_show()
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


def load_local_nq_data(split="train"):
    """Load dataset from local storage"""
    data_dir = "data/natural_questions"
    if not os.path.exists(data_dir):
        download_nq_data()

    dataset = DatasetDict.load_from_disk(data_dir)
    return dataset[split]


def preprocess_nq(example):
    """Extract first short answer from Natural Questions examples"""
    if example["annotations"][0]["short_answers"]:
        answer = example["annotations"][0]["short_answers"][0]["text"]
    else:
        answer = ""
    return {
        "id": example["id"],
        "question": example["question"]["text"],
        "context": " ".join(example["document"]["tokens"]),
        "answer": answer,
    }


def get_nq_data(split="train"):
    """Main data loading function with DVC integration"""
    try:
        return load_local_nq_data(split)
    except Exception as e:
        print(f"Error loading local data: {e}")
        download_nq_data()
        return load_local_nq_data(split)
