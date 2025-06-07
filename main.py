import time

import hydra
from omegaconf import DictConfig

from scripts.gpt_train import train_gpt
from scripts.vanilla_bert_benchmark import benchmark_bert
from scripts.vanilla_gpt_benchmark import benchmark_gpt
from utils.mps_enable import configure_mps


def run_benchmarks(bert_model_name, gpt_model_name, sample_size, batch_size):
    print("Benchmarking BERT Baseline")
    start = time.time()
    bert_loss = benchmark_bert(
        model_name=bert_model_name,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    print(f"BERT Validation Loss: {bert_loss}. Time: {time.time() - start:.0f}s")

    print("\nBenchmarking Vanilla GPT")
    start = time.time()
    gpt_loss = benchmark_gpt(
        model_name=gpt_model_name,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    print(f"GPT Validation Loss: {gpt_loss}. Time: {time.time() - start:.0f}s")
    return bert_loss, gpt_loss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    configure_mps()
    sample_size = cfg.data.val_sample_size

    # Initial benchmarks
    if cfg.run_type.benchmark:
        bert_loss, vanilla_gpt_loss = run_benchmarks(
            cfg.model.bert_name,
            cfg.model.gpt_name,
            cfg.data.val_sample_size,
            cfg.training.batch_size,
        )

    if cfg.run_type.train:
        print("\nTraining GPT Model")
        start = time.time()
        trained_model = train_gpt(
            model_name="gpt2",
            train_sample_size=2000,
            val_sample_size=sample_size,
            epochs=2,
        )
        print(f"Training completed. Time: {time.time() - start:.0f}s")

        print("\nBenchmarking Trained GPT")
        start = time.time()
        trained_loss = benchmark_gpt(trained_model, sample_size)
        print(f"Trained GPT Loss: {trained_loss} | Time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
