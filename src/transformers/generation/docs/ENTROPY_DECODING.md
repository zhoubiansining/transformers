# Entropy-Based Decoding — Technical Guide

## Overview

Entropy-based decoding selects the LLM layer closest to the "entropy valley" (the first layer where entropy stops decreasing) instead of always using the last layer. This can produce higher-quality tokens for certain tasks/models.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Basic usage: trough strategy (greedy backward scan for minimum entropy)
generation_config = GenerationConfig(
    do_sample=True,
    entropy_decoding="trough",   # use entropy valley layer for decoding
)
outputs = model.generate(inputs, generation_config=generation_config)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_decoding` | `str` or `None` | `None` | Decoding strategy: `"trough"` (entropy valley), `"random_after"` (random layer after trough), or `None` (observation-only). |
| `entropy_record_tokens` | `bool` or `None` | `None` | Record argmax token IDs per layer and selected token. |
| `entropy_record_per_layer_metrics` | `bool` or `None` | `None` | Record per-layer entropy and logits. |
| `entropy_record_per_layer_hidden_states` | `bool` or `None` | `None` | Record per-layer hidden states. |
| `entropy_logits_top_k` | `int` or `None` | `None` | Truncate per-layer logits to top-k values (reduces memory). |
| `entropy_offload_to_cpu` | `bool` or `None` | `True` | Offload recorded tensors to CPU non-blocking. |

## Usage Examples

### 1. Entropy Valley Decoding (Trough)

```python
generation_config = GenerationConfig(
    do_sample=True,
    entropy_decoding="trough",
)
outputs = model.generate(inputs, generation_config=generation_config)
```

### 2. Random Layer After Trough

```python
generation_config = GenerationConfig(
    do_sample=True,
    entropy_decoding="random_after",  # randomly pick a layer from trough to last
)
outputs = model.generate(inputs, generation_config=generation_config)
```

### 3. Observation Mode (No Decoding Change)

Record per-layer statistics without altering the decoding behavior (still uses last layer):

```python
generation_config = GenerationConfig(
    do_sample=True,
    entropy_record_per_layer_metrics=True,
    entropy_record_per_layer_hidden_states=True,
    entropy_logits_top_k=50,  # only keep top-50 logits per layer
    return_dict_in_generate=True,
)
outputs = model.generate(
    inputs,
    generation_config=generation_config,
)
# Access results via outputs.entropy_per_layer_entropies, etc.
```

### 4. Full Observation (All Flags)

```python
generation_config = GenerationConfig(
    do_sample=True,
    entropy_decoding="trough",
    entropy_record_tokens=True,
    entropy_record_per_layer_metrics=True,
    entropy_record_per_layer_hidden_states=True,
    entropy_logits_top_k=50,
    entropy_offload_to_cpu=True,
    return_dict_in_generate=True,
)
outputs = model.generate(
    inputs,
    generation_config=generation_config,
)
```

## Output Structure

When `return_dict_in_generate=True`, the output is a `GenerateDecoderOnlyOutput` object with:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `sequences` | `LongTensor` | `[B, T]` | Generated token IDs. |
| `entropy_selected_layer_indices` | `tuple` | `T * [B]` | Layer index (per step) where entropy reached its minimum. |
| `entropy_per_layer_entropies` | `tuple` | `T * [B, L]` | Normalized entropy per layer (0=first layer, L-1=last). |
| `entropy_per_layer_logits` | `tuple` | `T * [B, L, V]` | Full vocabulary logits per layer (only when `entropy_logits_top_k=None`). |
| `entropy_per_layer_top_k_logits` | `tuple` | `T * [B, L, K]` | Top-k logits per layer. |
| `entropy_per_layer_top_k_indices` | `tuple` | `T * [B, L, K]` | Top-k token indices per layer. |
| `entropy_per_layer_hidden_states` | `tuple` | `T * [B, L, H]` | Hidden states per layer. |
| `entropy_per_layer_token_ids` | `tuple` | `T * [B, L]` | Argmax token IDs per layer. |
| `entropy_selected_token_ids` | `tuple` | `T * [B]` | Token IDs from the selected (entropy valley) layer. |

**Note:** All entropy fields are **tuples** indexed by generation step (step 0 = prefill, step 1 = first decode, etc.).

## Entropy Value Interpretation

- Entropy is **normalized** to [0, 1]: 0 = deterministic (single token), 1 = uniform distribution.
- `entropy_selected_layer_indices[i]` = layer index (0-indexed) where entropy reached its minimum.
- A lower selected layer index means the model "stabilizes" earlier (good for simple tasks).
- A higher index means the model needs more layers to reach confidence (complex tasks).

## Memory Considerations

- **Default behavior**: `entropy_logits_top_k=None` records full vocabulary logits (`[B, L, V]`). This can be **very large** (e.g., ~1.6GB for B=1, L=40, V=32000 per step).
- **Recommended**: Set `entropy_logits_top_k=50` to truncate to top-50 tokens (~200KB/step).
- **CPU offload**: `entropy_offload_to_cpu=True` (default when any record flag is set) moves tensors to CPU via non-blocking transfer, freeing GPU memory quickly.

## Supported Models

Entropy valley decoding requires per-layer hidden states to be normalized with the **final LayerNorm/RMSNorm** before computing logits, ensuring a fair comparison across layers. The following models have this mechanism implemented:

- **Qwen3.5** (`Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`)
- **Qwen3.5 MoE** (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`)
- **GPT-OSS** (`GptOssForCausalLM`)
- **Gemma4** (`Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`)
- **DeepSeekV3** (`DeepseekV3ForCausalLM`)

For other models, the algorithm falls back to using pre-norm hidden states (which may produce suboptimal layer selection). Adding support for additional models requires adding a `get_final_norm()` method to the model's `ForCausalLM` / `ForConditionalGeneration` class that returns the final norm module.

## How It Works

At each generation step:

1. **Extract per-layer hidden states** from the model output (`outputs.hidden_states[1:]`, skipping the embedding output).
2. **Apply final normalization** to each layer's hidden state using the model's final RMSNorm/LayerNorm.
3. **Compute logits** via `lm_head(normalized_hidden)` for each layer.
4. **Calculate entropy** from the softmax distribution of each layer's logits.
5. **Select the entropy valley** — the first layer (from the back) where entropy stops decreasing.
6. **Decode** from the selected layer's logits (or last layer in observation-only mode).
