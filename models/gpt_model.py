from pathlib import Path

import dvc.api
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils.data_loader import get_nq_data as load_nq_data


def get_dvc_params():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"
    return dvc.api.params_show(str(config_path))


class GPT2QAModel(pl.LightningModule):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("avg_val_loss", avg_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)


def create_gpt_dataloader(dataset, tokenizer, batch_size=8):
    def tokenize_fn(examples):
        texts = [
            f"Question: {q} Answer: {a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        tokenized = tokenizer(
            texts,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return DataLoader(tokenized, batch_size=batch_size)


def benchmark_gpt(model_name="gpt2", sample_size=1000):
    model = GPT2QAModel(model_name)

    params = get_dvc_params()
    batch_size = params["model"]["batch_size"]
    sample_size = params["data"]["sample_size"]

    val_data = load_nq_data(split="validation", sample_size=sample_size)
    val_loader = create_gpt_dataloader(val_data, model.tokenizer, batch_size)

    trainer = pl.Trainer(
        accelerator="auto", devices="auto", logger=False, enable_checkpointing=False
    )
    results = trainer.validate(model, val_loader)
    return results[0]["avg_val_loss"].item()


def train_gpt(
    model_name="gpt2", train_sample_size=5000, val_sample_size=1000, epochs=3
):
    params = get_dvc_params()
    batch_size = params["model"]["batch_size"]
    epochs = params["model"]["epochs"]

    model = GPT2QAModel(model_name)
    train_data = load_nq_data(split="train", sample_size=train_sample_size)
    val_data = load_nq_data(split="validation", sample_size=val_sample_size)
    train_loader = create_gpt_dataloader(train_data, model.tokenizer, batch_size)
    val_loader = create_gpt_dataloader(val_data, model.tokenizer, batch_size)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        enable_checkpointing=True,
    )
    trainer.fit(model, train_loader, val_loader)
    return model
