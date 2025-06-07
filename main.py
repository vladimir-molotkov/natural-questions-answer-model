import time
from pathlib import Path

import dvc.api
import fire

from models.bert_model import benchmark_bert
from models.gpt_model import benchmark_gpt, train_gpt
from utils.mps_enable import configure_mps


def get_dvc_params():
    config_path = Path(__file__).parent / "configs" / "params.yaml"
    return dvc.api.params_show(str(config_path))


def run_benchmarks():
    params = get_dvc_params()
    sample_size = params["data"]["sample_size"]
    batch_size = params["model"]["batch_size"]

    print("Benchmarking BERT Baseline")
    start = time.time()
    bert_loss = benchmark_bert(sample_size, batch_size)
    print(f"BERT Validation Loss: {bert_loss:.3f}. Time: {time.time() - start:.0f}s")

    print("\nBenchmarking Vanilla GPT")
    start = time.time()
    gpt_loss = benchmark_gpt("gpt2", sample_size, batch_size)
    print(f"GPT Validation Loss: {gpt_loss:.3f}. Time: {time.time() - start:.0f}s")
    return bert_loss, gpt_loss


def main(benchmark=True, train=True):
    configure_mps()

    params = get_dvc_params()
    sample_size = params["data"]["sample_size"]

    # Initial benchmarks
    if benchmark:
        bert_loss, vanilla_gpt_loss = run_benchmarks()

    if train:
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
        print(
            f"Trained GPT Loss: {trained_loss:.4f} | Time: {time.time() - start:.0f}s"
        )

        improvement = (vanilla_gpt_loss - trained_loss) / vanilla_gpt_loss * 100
        print("\nSummary")
        print(f"BERT Loss: {bert_loss:.4f}")
        print(f"Vanilla GPT Loss: {vanilla_gpt_loss:.4f}")
        print(f"Trained GPT Loss: {trained_loss:.4f}")
        print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
