import time
from typing import Optional

import fire
import hydra
from omegaconf import OmegaConf

from scripts.gpt_train import train_gpt
from scripts.vanilla_bert_benchmark import benchmark_bert
from scripts.vanilla_gpt_benchmark import benchmark_gpt
from utils.mps_enable import configure_mps


def run_benchmarks(bert_model_name, gpt_model_name, sample_size, batch_size):
    start = time.time()
    bert_loss = benchmark_bert(
        model_name=bert_model_name,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    print(f"BERT Validation Loss: {bert_loss}. Time: {time.time() - start:.0f}s")

    start = time.time()
    gpt_loss = benchmark_gpt(
        model_name=gpt_model_name,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    print(f"GPT Validation Loss: {gpt_loss}. Time: {time.time() - start:.0f}s")
    return bert_loss, gpt_loss


def main(benchmark: Optional[bool] = True, train: Optional[bool] = True):
    configure_mps()

    with hydra.initialize(config_path="configs"):
        hydra_cfg = hydra.compose(config_name="config")

    cli_conf = OmegaConf.create(
        {
            "run_type": {
                "benchmark": benchmark,
                "train": train,
            }
        }
    )
    cfg = OmegaConf.merge(hydra_cfg, cli_conf)
    print("Config finished")

    sample_size = cfg.data.val_sample_size
    if cfg.run_type.benchmark:
        bert_loss, vanilla_gpt_loss = run_benchmarks(
            cfg.model.bert_name,
            cfg.model.gpt_name,
            cfg.data.val_sample_size,
            cfg.training.batch_size,
        )
        print(f"BERT Loss: {bert_loss}, GPT Loss {vanilla_gpt_loss}")

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

        start = time.time()
        trained_loss = benchmark_gpt(trained_model, sample_size)
        print(f"Trained GPT Loss: {trained_loss} | Time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    fire.Fire(main())
