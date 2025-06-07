from typing import Optional

import fire
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2QAInference:
    def __init__(
        self, checkpoint_path: Optional[str] = "", device: Optional[str] = "mps"
    ):
        """
        Load trained GPT-2 model for question answering

        Args:
            checkpoint_path: Path to trained PyTorch Lightning checkpoint
            device: "mps", "cpu", or "cuda"
        """
        self.device = torch.device(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if checkpoint_path:
            print("Loading trained model")
            self.model = GPT2LMHeadModel.from_pretrained(
                "gpt2",
                state_dict=torch.load(checkpoint_path, map_location=self.device)[
                    "state_dict"
                ],
            )
        else:
            print("Using GPT-2 from Hugging Face")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        self.model.eval()
        self.model.to(self.device)

    def ask(self, question, max_length=100, temperature=0.7, top_k=50):
        """
        Ask a question and get an answer from the model

        Args:
            question: Your question string
            max_length: Maximum answer length
            temperature: Sampling temperature (0.1-1.0)
            top_k: Top-k sampling parameter
        """
        prompt = f"Question: {question} Answer:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                no_repeat_ngram_size=2,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text.split("Answer:")[1].strip()

        if answer.endswith("?") or answer.endswith("?"):
            answer = answer[:-1]
        if "\n" in answer:
            answer = answer.split("\n")[0]

        return answer


def main(checkpoint_path, device="auto"):
    """
    Interactive QA session with trained GPT-2 model

    Args:
        checkpoint_path: Path to trained model checkpoint
        device: "auto", "mps", "cpu", or "cuda"
    """
    qa_system = GPT2QAInference(checkpoint_path, device)
    print("\nQA model Ready\n")

    while True:
        question = input("Your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = qa_system.ask(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    fire.Fire({"interactive": main, "single": GPT2QAInference})
