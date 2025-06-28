# 🧠 Fine-Tuning Gemma 3B on Healthcare Admin Tasks

This repository demonstrates how to fine-tune the instruction-tuned [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) model on a custom dataset covering administrative tasks in the healthcare industry.

---

## 📌 Project Overview

We use the [Unsloth](https://github.com/unslothai/unsloth) framework to:
- Load and quantize the base Gemma model in **4-bit precision**.
- Apply **LoRA (Low-Rank Adaptation)** for efficient parameter tuning.
- Train the model using Hugging Face's `trl` library and `SFTTrainer`.

This setup significantly reduces memory footprint and training cost, making it suitable for training on consumer GPUs (e.g. Colab, T4, A100).

---

## 🩺 Dataset: Healthcare Admin

- **Source**: [`xgalaxy/healthcare_admin`](https://huggingface.co/datasets/xgalaxy/healthcare_admin)
- **Format**: ShareGPT-style JSON format with structured `user` and `assistant` roles
- **Coverage**:
  - Appointment scheduling, cancellation, and rescheduling
  - Edge cases involving follow-ups, missing info, and ambiguous requests
  - Multi-turn conversations to emulate real-world interactions

---

## 🛠️ Key Components

### ✅ Model Setup
- `google/gemma-3-4b-it` loaded using Unsloth's `FastModel.from_pretrained()`
- 4-bit quantization enabled via `load_in_4bit=True`
- LoRA adapters injected for memory-efficient tuning

### ✅ Training
- Supervised fine-tuning with `SFTTrainer`
- Batch size simulated using `gradient_accumulation_steps`
- Linear learning rate scheduler with warmup
- Training capped at a fixed number of steps for fast iteration

---

## 🚀 Trained Model

The fine-tuned model is available on Hugging Face:
👉 [xgalaxy/gemma-3](https://huggingface.co/xgalaxy/gemma-3)

---

## 🔗 Resources

- 🔗 [Unsloth GitHub](https://github.com/unslothai/unsloth)
- 📘 [Gemma on Hugging Face](https://huggingface.co/google/gemma-3-4b-it)
- 🗃️ [Healthcare Admin Dataset](https://huggingface.co/datasets/xgalaxy/healthcare_admin)

---
