# GRPO-Unlearn: Selective Unlearning in LLMs via GRPO

This project implements a novel approach to machine unlearning in Large Language Models (LLMs) using Group Relative Policy Optimization (GRPO). The key innovation is the integration of another LLM as the objective function within the GRPO framework, creating a more nuanced and adaptive unlearning process.

## Project Structure

- `unlearning_pipeline.py`: Main script implementing the GRPO-based unlearning pipeline
- `evaluate.py`: Script for evaluating model performance before and after unlearning
- `requirements.txt`: Project dependencies
- `README.md`: This file

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Running the Unlearning Pipeline

To run the unlearning pipeline:

```bash
python unlearning_pipeline.py
```

This will:
1. Load the base model (Llama-2-1b by default)
2. Prepare training data from the ToxiGen dataset
3. Train the model using GRPO with an LLM judge
4. Save the unlearned model

### Evaluating the Model

To evaluate the model's performance:

```bash
python evaluate.py
```

This will:
1. Evaluate the original model
2. Evaluate the unlearned model
3. Save results to the `results/` directory
4. Print a comparison of the results

## Configuration

You can modify the following parameters in the scripts:

- Model size and architecture
- Training hyperparameters (learning rate, batch size, etc.)
- Evaluation metrics and datasets
- Judge LLM configuration

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key for the judge LLM
- Sufficient disk space for model storage
