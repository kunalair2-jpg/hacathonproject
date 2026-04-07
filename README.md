---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# 📧 OpenEnv AI Email Triage & Response System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kunalair2-jpg/hacathonproject/blob/main/examples/grpo_training.ipynb)

![Reward Curve](https://img.shields.io/badge/Reward_Curve-Optimizing...-success)

## Motivation
Email triage is a universally understood, ubiquitous real-world problem. A highly competent AI must demonstrate **multi-step reasoning** (understanding the context and priority), **workflow actions** (archiving or escalating), and **generative capabilities** (drafting replies). This environment acts as a robust benchmark for these realistic multi-modal agentic paths, specifically targeting real-world utility over simple toy tasks.

## Environment Dimensions

### 🧠 Action Space (Dict)
The agent passes actions iteratively via predefined structures:
- `action_type`: `classify`, `reply`, `archive`, `escalate`
- `priority_label`: The predicted label (`spam`, `urgent`, `normal`) used during classification.
- `content`: Generated body string, enforced during the `reply` action.

### 👁 Observation Space (Object)
The observation yields immediate context and necessary meta-context:
- `current_email`: Structured email properties including body, sender, and thread associations.
- `step_count`: The number of steps the agent has taken (used to measure efficiency).
- `inbox_remaining`: Total items left.
- `thread_memory`: Temporal context tracking recently processed thread paths, simulating human working memory.

### 🎯 Reward Design & Shaping
The environment implements **Dense Partial Rewards**, avoiding the trap of binary outcomes:
- **+0.3** - Correctly classifying the intent.
- **+0.5** - Selecting the correct workflow action (e.g., escalating an urgent item).
- **+0.2** - Generating a detailed, high-quality reply.
- **-0.01 per step** - A compounding efficiency penalty ensures the agent learns to prioritize processing speed and minimizes unnecessary looping or 'noop' sequences.

## Curriculum Learning Tasks
1. **Easy:** Strictly binary decisions (clear spam vs explicit meetings).
2. **Medium:** Ambiguous phrasing requiring sentiment and contextual analysis.
3. **Hard:** Multi-state dependencies requiring thread history retention and generative long-form capabilities.

## Execution

```bash
# Build the Environment
docker build -t email-env .

# Validate it natively with OpenEnv
openenv validate

# Test Inference
docker run email-env
```
