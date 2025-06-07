from models.bert_model import benchmark_bert
from models.gpt_model import benchmark_gpt, train_gpt
import time
from utils.mps_enable import configure_mps


def run_benchmarks(n_sample=1000):
    print("Benchmarking BERT Baseline")
    start = time.time()
    bert_loss = benchmark_bert(n_sample)
    print(f"BERT Validation Loss: {bert_loss:.3f}. Time: {time.time()-start:.0f}s")
    
    print("\nBenchmarking Vanilla GPT")
    start = time.time()
    gpt_loss = benchmark_gpt("gpt2", n_sample)
    print(f"GPT Validation Loss: {gpt_loss:.3f}. Time: {time.time()-start:.0f}s")
    return bert_loss, gpt_loss

def main():
    configure_mps()

    sample_size = 1000
    
    # Initial benchmarks
    bert_loss, vanilla_gpt_loss = run_benchmarks()
    
    print("\nTraining GPT Model")
    start = time.time()
    trained_model = train_gpt(
        model_name="gpt2",
        train_sample_size=2000,
        val_sample_size=sample_size,
        epochs=2
    )
    print(f"Training completed. Time: {time.time()-start:.0f}s")
    
 
    print("\nBenchmarking Trained GPT")
    start = time.time()
    trained_loss = benchmark_gpt(trained_model, sample_size)
    print(f"Trained GPT Loss: {trained_loss:.4f} | Time: {time.time()-start:.1f}s")
    
  
    improvement = (vanilla_gpt_loss - trained_loss) / vanilla_gpt_loss * 100
    print(f"\nSummary")
    print(f"BERT Loss: {bert_loss:.4f}")
    print(f"Vanilla GPT Loss: {vanilla_gpt_loss:.4f}")
    print(f"Trained GPT Loss: {trained_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    main()