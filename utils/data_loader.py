from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch


class NaturalQuestionsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 8,
        max_length: int = 384,
        doc_stride: int = 128,
        num_workers: int = 4,
        split_size: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.num_workers = num_workers
        self.split_size = split_size

    def prepare_data(self):
        load_dataset("google-research-datasets/natural_questions")

    def setup(self, stage=None):
        dataset = load_dataset("google-research-datasets/natural_questions", split="train")
        
        # Only short answers
        dataset = dataset.filter(lambda x: len(x["annotations"][0]["short_answers"]) > 0)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        def preprocess(example):
            question = example["question_text"]
            context = example["document_text"]
            answers = example["annotations"][0]["short_answers"]

            start_token = answers[0]["start_token"]
            end_token = answers[0]["end_token"]
            context_tokens = context.split()
            short_answer = " ".join(context_tokens[start_token:end_token])

            char_start = context.find(short_answer)
            char_end = char_start + len(short_answer)

            tokenized = tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            samples = []

            for i in range(len(tokenized["input_ids"])):
                offsets = tokenized["offset_mapping"][i]
                input_ids = tokenized["input_ids"][i]
                attention_mask = tokenized["attention_mask"][i]

                start_position = 0
                end_position = 0
                sequence_ids = tokenized.sequence_ids(i)

                for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
                    if seq_id != 1:
                        continue
                    if offset[0] <= char_start < offset[1]:
                        start_position = idx
                    if offset[0] < char_end <= offset[1]:
                        end_position = idx

                samples.append({
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "start_positions": torch.tensor(start_position),
                    "end_positions": torch.tensor(end_position),
                })

            return samples

        tokenized_dataset = dataset.map(
            preprocess,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        self.tokenizer = tokenizer
        split = int(len(tokenized_dataset) * (1 - self.split_size))
        self.train_dataset = tokenized_dataset.select(range(split))
        self.val_dataset = tokenized_dataset.select(range(split, len(tokenized_dataset)))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )