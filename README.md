# LeRobot: Vision-Language Quadruped Control

This repository contains the codebase for **real-time natural-language control of quadruped robots** using Vision-Language-Action (VLA) models and offline reinforcement learning. It builds on [Hugging Face's `lerobot`](https://github.com/huggingface/lerobot) framework and includes tooling for simulation, data collection, dataset conversion, and training with both SmolVLA and IQL.

---

## 🛠 Setup

First, install the required dependencies:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
````

---

## 🚀 Training SmolVLA

Fine-tune the pretrained 450M parameter `smolvla_base` model on quadruped locomotion data:

```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=quad_data/processed_dataset \
  --dataset.root=quad_data/processed_dataset \
  --batch_size=64 \
  --steps=40000 \
  --log_freq=100 \
  --save_freq=1000 \
  --save_checkpoint=True \
  --eval_freq=2000 \
  --eval.n_episodes=25 \
  --eval.batch_size=25 \
  --num_workers=8
```

---

## 📦 Repository Structure

* `docker/isaac_sim/`: Dockerfiles and setup scripts for running Isaac Sim in containerized environments.
* `isaac_sim/`: Scripts to launch the Spot robot in simulation and collect demonstration trajectories with natural-language prompts.
* `quad_data/`: Utilities to convert raw Isaac Sim logs into:

  * LeRobot-compatible dataset format
  * Reward-annotated format for IQL

---

## 📁 Datasets

Please place all datasets under the `quad_data/` directory.

* 📷 **Raw simulation logs**:
  [Download](https://drive.google.com/drive/folders/1SM1VzLHcGzx6Kd-U8O5k2HIxsVvTcLW0?usp=drive_link)

* 📊 **LeRobot processed dataset** (for SmolVLA training):
  [Download](https://drive.google.com/drive/folders/1CfZ_uUwREcmZYeXMYpWWzOivjQWj84Pd?usp=drive_link)

* 🎯 **Reward-annotated dataset** (for IQL training):
  [Download](https://drive.google.com/drive/folders/1JJmkWaKAyty5rXCzdXXgerdADuWmb7eD?usp=drive_link)

---

## 🎓 Pretrained Models

You can download trained model checkpoints here:

* 🧠 **SmolVLA**:
  [Download](https://drive.google.com/drive/folders/1EDL0R6RuItyAtfddZqPGE2RRdaG6El21?usp=drive_link)

* 🧠 **TinyVLA** (initial experiments, not used in final eval):
  [Download](https://drive.google.com/drive/folders/1Jou3EsA9u_ipA5ojoRP5xOo0Y484NbKt?usp=drive_link)

* 🤖 **IQL (Implicit Q-Learning)**:
  [Download](https://drive.google.com/drive/folders/184R8LL9U6PluTmmyrvGTyu9ScXOU49XE?usp=drive_link)

---

## 📎 Citation

If you use this work in your research or projects, please cite the original SmolVLA paper and this repository.

---

## 📬 Contact

Maintainer: **Sagar Manglani**
Email: [sagarm@stanford.edu](mailto:sagarm@stanford.edu)
