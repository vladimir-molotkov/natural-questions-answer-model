import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast

from scripts.data_loader import get_nq_data
from utils.mlflow_utils import MLflowLogger


class BertQAModel(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.validation_step_outputs = []
        self.mlflow_logger = MLflowLogger()
        self.mlflow_logger.log_params(
            {
                "model_type": "BERT",
                "model_name": model_name,
                "task": "question_answering",
            }
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

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
        return torch.optim.Adam(self.parameters(), lr=5e-5)


def create_bert_dataloader(dataset, tokenizer, batch_size=8):
    def tokenize_fn(examples):
        return tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch")
    return DataLoader(tokenized, batch_size=batch_size)


def benchmark_bert(model_name, sample_size, batch_size):
    model = BertQAModel(model_name)
    tokenizer = model.tokenizer
    val_data = get_nq_data(split="validation", sample_size=sample_size)
    val_loader = create_bert_dataloader(val_data, tokenizer, batch_size)

    trainer = pl.Trainer(
        accelerator="auto", devices="auto", logger=False, enable_checkpointing=False
    )
    results = trainer.validate(model, val_loader)
    return results
