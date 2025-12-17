# Multimodal LLM – Three Implementation Paths

This repository demonstrates three common approaches used to build
Multimodal Large Language Models (MLLMs). The focus of this assignment
is on understanding system architecture rather than training large
models from scratch.

---

## Path 1: LLM + Tools
This approach connects independent pretrained models in a pipeline.
Speech input is first converted to text, then processed by a language
model, and finally used to generate an image.

**Input:** Audio file (speech)  
**Output:** Generated text and image

---

## Path 2: LLM + Adapters
In this approach, large pretrained models are kept frozen and small
trainable adapter layers are used to align representations from
different modalities.

**Input:** Image feature vector  
**Output:** Aligned feature representation for the LLM

---

## Path 3: Unified Multimodal Model
This approach uses a single model with shared representations for
text, image, and audio. All modalities are processed jointly using a
common transformer architecture.

**Input:** Text, image features, audio features  
**Output:** Joint multimodal representation

---

## Notes
The implementations are simplified and intended for academic learning.
They demonstrate architectural concepts rather than production-level
performance.

