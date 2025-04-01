import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import json
import os
from unlearning_pipeline import UnlearningPipeline, JudgeModel
import wandb


class ModelEvaluator:
    def __init__(
        self,
        model_name: str,
        judge_model: JudgeModel = JudgeModel.GPT4O,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = UnlearningPipeline(
            model_name=model_name,
            judge_model=judge_model,
        )

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_on_dataset(
        self,
        dataset_name: str = "microsoft/toxigen",
        split: str = "test",
        num_samples: int = 100,
    ) -> Dict:
        """Evaluate model on a dataset."""
        dataset = load_dataset(dataset_name, split=split)
        results = {
            "responses": [],
            "metrics": {
                "average_reward": 0.0,
                "unlearning_success_rate": 0.0,
                "response_lengths": [],
            },
        }

        total_reward = 0
        unlearning_successes = 0

        for item in tqdm(dataset[:num_samples], desc="Evaluating"):
            prompt = item["prompt"]
            response = self.generate_response(prompt)

            # Get reward from judge
            reward = self.pipeline.get_llm_judge_reward(
                prompt=prompt,
                response=response,
                unlearning_criteria=item.get(
                    "unlearning_criteria", "Remove harmful or biased content"
                ),
            )

            # Update metrics
            total_reward += reward
            if reward > 0.5:  # Consider it successful if reward > 0.5
                unlearning_successes += 1
            results["response_lengths"].append(len(response))

            results["responses"].append(
                {
                    "prompt": prompt,
                    "generated_response": response,
                    "ground_truth": item.get("response", ""),
                    "reward": reward,
                }
            )

        # Calculate final metrics
        results["metrics"]["average_reward"] = total_reward / num_samples
        results["metrics"]["unlearning_success_rate"] = (
            unlearning_successes / num_samples
        )
        results["metrics"]["avg_response_length"] = np.mean(results["response_lengths"])
        results["metrics"]["std_response_length"] = np.std(results["response_lengths"])

        return results

    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to a file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def log_to_wandb(self, results: Dict, model_name: str):
        """Log evaluation results to Weights & Biases."""
        wandb.log(
            {
                f"{model_name}/average_reward": results["metrics"]["average_reward"],
                f"{model_name}/unlearning_success_rate": results["metrics"][
                    "unlearning_success_rate"
                ],
                f"{model_name}/avg_response_length": results["metrics"][
                    "avg_response_length"
                ],
                f"{model_name}/std_response_length": results["metrics"][
                    "std_response_length"
                ],
            }
        )


def main():
    # Initialize wandb for experiment tracking
    wandb.init(project="grpo-unlearning-evaluation")

    # Evaluate original model
    print("Evaluating original model...")
    original_evaluator = ModelEvaluator("meta-llama/Llama-2-1b-hf")
    original_results = original_evaluator.evaluate_on_dataset(
        "microsoft/toxigen",
        split="test",
        num_samples=100,
    )
    original_evaluator.save_results(
        original_results, "results/original_model_results.json"
    )
    original_evaluator.log_to_wandb(original_results, "original_model")

    # Evaluate unlearned model
    print("Evaluating unlearned model...")
    unlearned_evaluator = ModelEvaluator("unlearned_model")
    unlearned_results = unlearned_evaluator.evaluate_on_dataset(
        "microsoft/toxigen",
        split="test",
        num_samples=100,
    )
    unlearned_evaluator.save_results(
        unlearned_results, "results/unlearned_model_results.json"
    )
    unlearned_evaluator.log_to_wandb(unlearned_results, "unlearned_model")

    # Compare results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Original Model:")
    print(f"  Average Reward: {original_results['metrics']['average_reward']:.3f}")
    print(
        f"  Unlearning Success Rate: {original_results['metrics']['unlearning_success_rate']:.3f}"
    )
    print(
        f"  Average Response Length: {original_results['metrics']['avg_response_length']:.1f}"
    )
    print(f"\nUnlearned Model:")
    print(f"  Average Reward: {unlearned_results['metrics']['average_reward']:.3f}")
    print(
        f"  Unlearning Success Rate: {unlearned_results['metrics']['unlearning_success_rate']:.3f}"
    )
    print(
        f"  Average Response Length: {unlearned_results['metrics']['avg_response_length']:.1f}"
    )


if __name__ == "__main__":
    main()
