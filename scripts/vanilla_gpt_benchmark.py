import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from scripts.data_loader import get_nq_data
from utils.mlflow_utils import MLflowLogger


class GPT2QAModel(pl.LightningModule):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.validation_step_outputs = []
        self.mlflow_logger = MLflowLogger()
        self.mlflow_logger.log_params(
            {
                "model_type": "BERT",
                "model_name": model_name,
                "task": "question_answering",
            }
        )

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
    tokenized.set_format(type="torch")
    return DataLoader(tokenized, batch_size=batch_size)


def benchmark_gpt(model_name="gpt2", sample_size=1000, batch_size=8):
    model = GPT2QAModel(model_name)
    val_data = get_nq_data(split="validation", sample_size=sample_size)
    val_loader = create_gpt_dataloader(val_data, model.tokenizer, batch_size)

    trainer = pl.Trainer(
        accelerator="auto", devices="auto", logger=False, enable_checkpointing=False
    )
    results = trainer.validate(model, val_loader)
    return results
