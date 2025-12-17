# Multimodal LLM Implementations

This repository contains simple implementations of three common
approaches to building Multimodal Large Language Models (MLLMs).

The goal of this project is to understand architectural ideas rather
than train large-scale models.

## Path 1: LLM + Tools
This approach connects independent pretrained models in a pipeline.
For example, speech is converted to text, processed by an LLM, and
then converted to an image.

## Path 2: LLM + Adapters
In this approach, large pretrained models are frozen and small
trainable adapter layers are added to align different modalities.

## Path 3: Unified Multimodal Model
This approach uses a single model with shared representations for
text, image, and audio. The implementation here is a simplified
version to demonstrate the concept.

These implementations are intended for academic learning purposes.

