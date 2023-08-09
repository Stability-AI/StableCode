# StableCode: Stability AI Developer Productivity/Developer


This repository contains Stability AI's ongoing development of the StableCode series of code models and will be continuously updated with new checkpoints. The following provides an overview of all currently available models. More coming soon.

## News


*August 8, 2023*

- Released the initial suite of StableCode-Alphas. Catch the release blog post [here](https://stability.ai/blog/stablecode-llm-generative-ai-coding). Which includes,
-  

*August 9, 2023*
- Released 

## Models

### StableCode-Completion-Alpha v2

StableLM-Alpha v2 models significantly improve on the initial Alpha models by incorporating architectural improvements such as SwiGLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)) and using higher-quality data sources, as discussed below.  The context length for these models is 4096 tokens.

| Size | StableLM-Base-Alpha-v2                                                     | Training Tokens | Parameters    |
|------|----------------------------------------------------------------------------|-----------------|---------------|
| 3B   | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-3b-v2) | 1.1T            | 2,796,431,360 |
| 7B   | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-7b-v2) | 1.1T            | 6,890,209,280 |

#### Training Details

Please refer to the provided YAML configuration files for hyperparameter details. E.g. for the extended `StableLM-Alpha-3B-v2` model, see [stablelm-base-alpha-3b-v2-4k-extension.yaml](./configs/stablelm-base-alpha-3b-v2-4k-extension.yaml).

Following similar work, we use a multi-stage approach to context length extension ([Nijkamp et al., 2023](https://blog.salesforceairesearch.com/xgen/)), scheduling 1 trillion tokens at context length 2048 followed by 100 billion tokens at 4096. We found that sequence length warmup ([Li et al., 2022](https://arxiv.org/abs/2108.06084)) helped stabilize early spikes during the first ~80 billion tokens of pre-training. However, it was not applied to the final runs due to significant throughput penalties as length shapes grew across the curriculum.

#### Training Data

The most impactful changes for StableLM-Alpha-v2 downstream performance were in the usage of higher quality data sources and mixtures; specifically, the use of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) and [C4](https://huggingface.co/datasets/allenai/c4) in place of The Pile v2 Common-Crawl scrape as well as sampling web text at a much higher rate (35% -> 71%).

The first pre-training stage relies on 1 trillion tokens sourced from a mix of the public Falcon RefinedWeb extract ([Penedo et al., 2023](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)), RedPajama-Data ([Together Computer., 2023](https://github.com/togethercomputer/RedPajama-Data)), The Pile ([Gao et al., 2020](https://arxiv.org/abs/2101.00027)), and internal datasets with web text sampled at a rate of 71%.

In the second stage, we include the StarCoder ([Li et al., 2023](https://arxiv.org/abs/2305.06161)) dataset and down sample web text to 55% while increasing sampling proportions of naturally long text examples in the aforementioned sources.

#### Evaluation

The following zero-shot evaluations are performed with the `lm-evaluation-harness` at commit [`df3da98c5405deafd519c2ddca52bb7c3fe36bef`](https://github.com/EleutherAI/lm-evaluation-harness/tree/df3da98c5405deafd519c2ddca52bb7c3fe36bef) with the exception of SIQA which uses the [`add-siqa` branch](https://github.com/EleutherAI/lm-evaluation-harness/tree/add-siqa) with prompt format
`{doc['context']}\nQuestion: {doc['question']}\nAnswer:`.

| Model                     | ARC Challenge✱ | ARC Easy✱ | BoolQ | HellaSwag✱ | LAMBADA<br>OpenAI | OpenBookQA | PIQA  | SIQA  | TruthfulQA▲ | Winogrande | Average |
| ------------------------- |:---------------:|:----------:|:-----:|:-----------:|:-----------------:|:----------:|:-----:|:-----:|:------------:|:----------:|:-------:|
| **StableLM-Alpha-7B-v2** | 40.53           | 69.11      | 70.31 | 74.27       | 74.19             | 30.40      | 78.45 | 42.43 | 36.46        | 68.82      | 58.50   |
| LLaMA-2-7B                | 46.16           | 74.54      | 77.74 | 75.94       | 73.47             | 31.40      | 77.75 | 43.50 | 38.97        | 69.61      | 60.91   |
| MPT-7B                    | 41.89           | 70.03      | 73.94 | 76.17       | 68.64             | 31.40      | 78.89 | 45.14 | 33.49        | 68.03      | 58.76   |
| OpenLLaMA-7B-v2           | 42.41           | 69.65      | 71.41 | 74.65       | 71.05             | 30.20      | 79.16 | 41.97 | 34.57        | 65.82      | 58.09   |
| RedPajama-INCITE-7B-Base  | 39.42           | 69.19      | 70.76 | 70.33       | 71.34             | 29.00      | 77.15 | 42.58 | 33.01        | 64.33      | 56.71   |
| **StableLM-Alpha-3B-v2** | 35.07           | 63.26      | 64.56 | 68.58       | 70.25             | 26.40      | 76.01 | 42.48 | 35.87        | 62.12      | 54.46   |
| BTLM-3B-8K           | 37.63           | 67.09      | 69.63 | 69.78       | 66.23             | 27.60      | 75.84 | 42.78 | 36.00        | 64.96      | 55.75   |
| OpenLLaMA-3B-v2           | 36.09           | 63.51      | 65.69 | 69.99       | 66.74             | 26.00      | 76.66 | 41.20 | 34.59        | 62.90      | 54.34   |
| Pythia-2.8B (deduped)     | 32.94           | 59.09      | 64.13 | 59.44       | 65.15             | 23.80      | 74.10 | 40.94 | 35.56        | 58.25      | 51.34   |
| StableLM-Alpha-7B    | 27.05           | 44.87      | 60.06 | 41.22       | 55.11             | 21.40      | 66.76 | 39.46 | 39.96        | 50.12      | 44.60   |
| StableLM-Alpha-3B    | 25.77           | 42.05      | 57.65 | 38.31       | 41.72             | 17.00      | 63.82 | 35.62 | 40.53        | 52.64      | 41.51   |

✱: Denotes byte-length normalized accuracy (`acc_norm`) as described in [Gao, 2021](https://blog.eleuther.ai/multiple-choice-normalization/).

▲: We score TruthfulQA using the normalized total probability assigned to the set of true answers (`mc2`).


## Quickstart

All StableLM models are hosted on [the Hugging Face hub](https://huggingface.co/StabilityAI). Check out this [notebook](https://github.com/Stability-AI/StableLM/blob/main/notebooks/stablelm-alpha.ipynb) to run inference with limited GPU capabilities.

Get started chatting with `StableLM-Tuned-Alpha` by using the following code snippet:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
model.half().cuda()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.7,
  do_sample=True,
  stopping_criteria=StoppingCriteriaList([StopOnTokens()])
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```


## Request for Help

Want to get involved?

- We would love to port [llama.cpp](https://github.com/ggerganov/llama.cpp) to work with StableLMs
- Integration into [Open Assistant](https://github.com/LAION-AI/Open-Assistant) from LAION-AI to collect high quality human-generated feedback data
- ... Reach out to us with ideas on our [Discord](https://discord.com/invite/stablediffusion)

## Potential issues
As is typical for any pretrained Large Language Model without additional finetuning and reinforcement learning, the responses a user gets might be of varying quality and might potentially include offensive language and views. This is expected to be improved with scale, better data, community feedback, and optimisation.

## Acknowledgements

- `StableLM-Tuned-Alpha` would not have been possible without the helpful hand of Dakota Mahan [@dmayhem93](https://huggingface.co/dmayhem93).

## Licenses

- Base model checkpoints (`StableLM-Base-Alpha`) are licensed under the Creative Commons license ([CC BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Under the license, you must give [credit](https://creativecommons.org/licenses/by/4.0/#) to Stability AI, provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by/4.0/#). You may do so in any reasonable manner, but not in any way that suggests the Stability AI endorses you or your use.

- Fine-tuned checkpoints (`StableLM-Tuned-Alpha`) are licensed under the Non-Commercial Creative Commons license ([CC BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)), in-line with the original non-commercial license specified by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

- All code in this repository is licensed under the Apache License 2.0 license.
