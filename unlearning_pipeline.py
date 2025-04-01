import os
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import wandb
from enum import Enum


class JudgeModel(Enum):
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O3_MINI = "o3-mini"

    @classmethod
    def get_all_models(cls) -> List[str]:
        return [model.value for model in cls]


class UnlearningPipeline:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B",
        judge_model: JudgeModel = JudgeModel.GPT4O,
        openai_api_key: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        # Enable TF32 for better performance on A100
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Llama 3.2 uses bfloat16 by default
            device_map="auto",
            # Enable gradient checkpointing for memory efficiency
            gradient_checkpointing=True,
            # Llama 3.2 specific configurations
            use_cache=True,
            trust_remote_code=True,
            # Add model-specific configurations
            max_position_embeddings=128000,  # Llama 3.2 supports 128k context
            rope_scaling={
                "type": "linear",
                "factor": 2.0,
            },  # For better long context handling
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",  # Important for causal language modeling
        )
        self.judge_model = judge_model.value  # Get the string value from the enum

        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        elif "OPENAI_API_KEY" in os.environ:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError("OpenAI API key not provided")

        # Set up the prompt template for Llama 3.2
        self.prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant that has been trained to unlearn harmful or biased content.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}"""

    def get_llm_judge_reward(
        self, prompt: str, response: str, unlearning_criteria: str
    ) -> float:
        """Get reward from LLM judge based on unlearning criteria."""
        judge_prompt = f"""
        You are a response evaluator that checks whether the response contains information that should be unlearned. The goal is to use this information to make a model unlearn informatino relating to a given criteria. You will be provided with a prompt, a response, and an unlearning criteria, and you are tasked with responding in JSON that details whether the response should be unlearned or not.

        JSON FORMAT: {{
            "reasoning": str,
            "should_unlearn": bool,
        }}

        Here is the prompt, response, and unlearning criteria:
        Prompt: {prompt}        
        Response: {response}
        Unlearning Criteria: {unlearning_criteria}"""

        try:
            completion = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "developer", "content": judge_prompt},
                ],
                temperature=0.0 if self.judge_model != "gpt-3.5-turbo-0125" else None,
            )
            return float(completion.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error getting judge reward: {e}")
            return 0.0

    def prepare_training_data(
        self, dataset_name: str, split: str = "train"
    ) -> List[Dict]:
        """Prepare training data from a dataset."""
        dataset = load_dataset(dataset_name, split=split)
        return [
            {
                "prompt": item["prompt"],
                "response": item["response"],
                "unlearning_criteria": item.get(
                    "unlearning_criteria", "Remove harmful or biased content"
                ),
            }
            for item in dataset
        ]

    def train(
        self,
        training_data: List[Dict],
        num_epochs: int = 3,
        batch_size: int = 8,  # Increased batch size for A100
        learning_rate: float = 2e-5,
        output_dir: str = "unlearned_model",
    ):
        """Train the model using GRPO for unlearning."""
        trainer = GRPOTrainer(
            model=self.model,
            args={
                "output_dir": output_dir,
                "num_train_epochs": num_epochs,
                "per_device_train_batch_size": batch_size,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": 2,  # Reduced for A100
                "save_strategy": "epoch",
                # A100-specific optimizations
                "fp16": True,
                "gradient_checkpointing": True,
                "optim": "adamw_torch_fused",  # Use fused optimizer for better performance
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "logging_steps": 10,
                "evaluation_strategy": "steps",
                "eval_steps": 100,
            },
            train_dataset=training_data,
            tokenizer=self.tokenizer,
            reward_fn=lambda x: self.get_llm_judge_reward(
                x["prompt"], x["response"], x["unlearning_criteria"]
            ),
        )

        trainer.train()
        trainer.save_model(output_dir)


def main():
    # Initialize wandb for experiment tracking
    wandb.init(project="grpo-unlearning")

    # Initialize pipeline
    pipeline = UnlearningPipeline()

    # Example usage with ToxiGen dataset
    training_data = pipeline.prepare_training_data("microsoft/toxigen", split="train")

    # Train the model
    pipeline.train(
        training_data=training_data,
        num_epochs=3,
        batch_size=8,  # Increased for A100
        learning_rate=2e-5,
        output_dir="unlearned_model",
    )


if __name__ == "__main__":
    main()
