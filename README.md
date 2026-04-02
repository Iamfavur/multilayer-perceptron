# Multi-Layer Perceptron (Part 2 of Language Modeling)

This repository focuses on building and training a character-level **Multi-Layer Perceptron (MLP)** for name generation using PyTorch.

It is a direct continuation of the Intro to Language Modeling work, but this repo is intentionally centered on **Part 2 (the MLP model)**.

## Project Context

This codebase assumes you already saw the baseline language modeling ideas in Part 1:

- character-level tokenization
- context windows for next-character prediction
- probability-based generation

Part 1 has its own repository. This repository starts from that foundation and moves into a neural-network approach with trainable embeddings and hidden layers.

## Main Focus: `multi_layer_perceptron_part2.ipynb`

The core notebook in this repo is `multi_layer_perceptron_part2.ipynb`. It walks through the full MLP pipeline step by step:

1. Load and inspect the dataset from `names.txt`
2. Build character vocabulary mappings (`stoi` / `itos`)
3. Construct training examples using a fixed context size (`block_size = 3`)
4. Split data into train/dev/test sets (80/10/10)
5. Create a learnable embedding table (`C`) for characters
6. Define an MLP with:
   - flattened context embeddings
   - hidden layer + `tanh` activation
   - output projection to vocabulary logits
7. Train with mini-batch gradient descent
8. Use `F.cross_entropy` for stable classification loss
9. Apply learning-rate decay during training
10. Evaluate on dev/test and sample new names from the model

In short: this notebook shows how a basic count-based/bigram-style intuition evolves into a trainable neural language model.

## Repository Files

- `multi_layer_perceptron_part2.ipynb`: Main implementation and experiments for the MLP language model
- `intro_to_language_modeling_part1.ipynb`: Companion reference notebook (intro material), included here for continuity
- `names.txt`: Training corpus (list of names)
- `README.md`: Project documentation

## Learning Goals

By working through this notebook, you will practice:

- converting raw text into supervised next-token prediction examples
- implementing embeddings from scratch as trainable parameters
- understanding tensor shapes through forward passes
- training neural models with backpropagation and minibatches
- monitoring loss and validating to reduce overfitting
- sampling from model probabilities to generate realistic outputs

## How to Run

```bash
# from the repository root in VS-Code
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch matplotlib jupyter
```

Select the **venv** kernel

Then open `multi_layer_perceptron_part2.ipynb` in VS-code and run the cells in order.

## Notes

- This is an educational, notebook-first codebase meant to show model-building fundamentals clearly.
- The focus is understanding how MLP-based character language models work end to end, not production packaging.
