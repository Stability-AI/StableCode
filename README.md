# StableCode: Stability AI Developer Productivity/Developer


This repository contains Stability AI's ongoing development of the StableCode series of code models and will be continuously updated with new checkpoints. The following provides an overview of all currently available models. More coming soon.

## News


*August 8, 2023*

Released the initial suite of StableCode-Alphas. Catch the release blog post [here](https://stability.ai/blog/stablecode-llm-generative-ai-coding). Which includes,
- StableCode-Completion-Alpha-3B
- StableCode-Completion-Alpha-3B-4k
- StableCode-Instruct-Alpha-3b

*August 9, 2023*
- Released an improved version of StableCode-Completion-Alpha-3B -> [StableCode-Completion-Alpha-3B-v1.1](stabilityai/stablecode-completion-alpha-3b). This model is trained on more tokens of the top 5 languages.

## Models

### StableCode-Completion-Alpha suite of Models

-  [StableCode-Completion-Alpha-3B](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b) - `StableCode-Completion-Alpha-3B` is a 3 billion parameter decoder-only code completion model pre-trained on a diverse set of programming languages that were the top used languages based on the 2023 stackoverflow developer survey with a context length of 16k. Trained on a special augmented version of the [starcoder-dataset](https://huggingface.co/datasets/bigcode/starcoderdata/viewer/bigcode--starcoderdata/train?row=0).
- [StableCode-Completion-Alpha-3B-4k](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k) -  StableCode-Completion-Alpha-3B-4K is a 3 billion parameter decoder-only code completion model pre-trained on diverse set of programming languages that topped the stackoverflow developer survey with a context length of 4k.



#### Training Details

Following similar work, we use a multi-stage approach to context length extension ([Nijkamp et al., 2023](https://blog.salesforceairesearch.com/xgen/)), scheduling 390 billion tokens at context length 4096 followed by 100 billion tokens at 16k tokens. We found that sequence length warmup ([Li et al., 2022](https://arxiv.org/abs/2108.06084)) helped stabilize early spikes during the first ~80 billion tokens of pre-training. However, it was not applied to the final runs due to significant throughput penalties as length shapes grew across the curriculum.

#### Training Data
The training is done in two stages, initial pretraining with the top 12 languages, which we got inspired by Stackoverflow developer survey,
- java
- javascript
- python
- typescript,
- php
- sql
- rust
- c
- markdown,
- go
- c++
- Shell
  
This is then followed by continued pretraining with top 6 languages to be an expert in those languages,
- Java
- Javascript
- Python
- C
- C++
- Go
#### Evaluation

The following zero-shot evaluations are performed with the awesome [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness),   



| Name                                                                                                               | HuggingFace Name                              | Type              |    | Context Length | Human-Eval |
| ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- | ----------------- | -- | -------------- | ---------- |
|                                                                                                                    |                                               |                   |    |                | pass@1     | pass@10 |
| [StableCode-Completion-Alpha-3B](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b)                | stabilityai/stablecode-completion-alpha-3b    | Base              | 3B | 16384          | 20.18      | 33.75 |
| [StableCode-Completion-Alpha-3b-4k](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k/)         | stabilityai/stablecode-completion-alpha-3b-4k | Base              | 3B | 4096           | 17.68      | 27.01 |
| [Stablecode-Instruct-Alpha-3b](https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/)                   | stabilityai/stablecode-instruct-alpha-3b      | Instruction Tuned | 3B | 4096           | 26.89      | 36.18 |
| [StableCode-Completion-Alpha-3B v1.1](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b/tree/v1.1) | stabilityai/stablecode-instruct-alpha-3b      | Base              | 3B | 16384          | 22.06      | 33.37 |



   
## Quickstart

All StableCode models are hosted on [the Hugging Face hub](https://huggingface.co/StabilityAI). Check out this [notebook](https://github.com/Stability-AI/StableLM/blob/main/notebooks/stablelm-alpha.ipynb) to run inference with limited GPU capabilities.

Get started chatting with `StableCode-Completion-Alpha` by using the following code snippet:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablecode-completion-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablecode-completion-alpha-3b")
model.half().cuda()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids


prompt = f"import torch\nimport torch.nn as nn"

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

## Licenses

- Base model checkpoints (`StableCode-Completion-Alpha-3B`) are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

- Instruct-tuned checkpoints (`StableCode-Instruct-Alpha-3B`) are licensed under [StableCode Research License](https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/blob/main/LICENSE.md) Copyright (c) Stability AI Ltd. All Rights Reserved

- All code in this repository is licensed under the Apache License 2.0 license.
