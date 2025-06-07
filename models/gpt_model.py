import torch
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from utils.data_loader  import load_nq_data

class GPT2QAModel(pl.LightningModule):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

def create_gpt_dataloader(dataset, tokenizer, batch_size=8):
    def tokenize_fn(examples):
        texts = [f"Q: {q} A: {a[0] if a else ''}" for q, a in zip(examples["question"], examples["short_answers"])]
        tokenized = tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=batch_size)

def train_gpt():
    model = GPT2QAModel()
    train_data = load_nq_data(split="train", sample_size=10000)
    val_data = load_nq_data(split="validation", sample_size=1000)
    train_loader = create_gpt_dataloader(train_data, model.tokenizer)
    val_loader = create_gpt_dataloader(val_data, model.tokenizer)
    
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)