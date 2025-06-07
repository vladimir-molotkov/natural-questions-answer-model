from datasets import load_dataset


def preprocess_nq(example):
    """Extract short answers from Natural Questions examples"""
    short_answers = []
    for annotation in example["annotations"]:
        for short_answer in annotation["short_answers"]:
            if short_answer["text"]:
                short_answers.append(short_answer["text"])
    return {
        "id": example["id"],
        "question": example["question"]["text"],
        "context": " ".join(example["document"]["tokens"]),
        "short_answers": short_answers,
    }


def load_nq_data(split="train", sample_size=1000):
    """Load google-research-datasets/natural_questions dataset

    Args:
        split (str, optional): dataset part. Defaults to "train".
        sample_size (int, optional): sample size. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    dataset = load_dataset("google-research-datasets/natural_questions", split=split)
    dataset = dataset.map(preprocess_nq, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["short_answers"]) > 0)
    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    return dataset
