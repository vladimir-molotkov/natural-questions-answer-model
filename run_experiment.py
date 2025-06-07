from models.bert_model import benchmark_bert
from models.gpt_model import train_gpt
import numpy as np

if __name__ == "__main__":
    print("=== Benchmarking BERT Baseline ===")
    bert_loss = benchmark_bert()
    print(f"BERT Validation Loss: {bert_loss:.4f}")
    
    print("\n=== Training GPT Model ===")
    train_gpt()
    
    print("\n=== Evaluation Complete ===")