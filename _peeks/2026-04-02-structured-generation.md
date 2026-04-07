---
title: "Structured outputs from large language models"
date: 2026-04-02 15:13 +100
excerpt: How to make a LLM generate structured output with great reliability
category: peek generativeai coding llm
---

Large language models do not always return the same answer to a given question. This is a key feature called non-determinism, which makes conversations feel more natural. However, it prevents these models to be integrated with other systems that expect clear formats.

## The problem with non-determinism

A large language model will provide a slightly different answer to the same question. This is a feature, not a bug. These models generate their answers by sampling from a distribution of tokens (*nd: tokens are groups of characters that make the model's vocabulary*). Relevant tokens get a higher probability, however there is no guarantee that they will be picked.

<p align="center">
  <img src="/assets/images/token_sampling.png" alt="Alt text" style="max-width: 100%;">
  <em style="font-size: 0.8em;">Illustration of token sampling</em>
</p>

Other systems may benefit from large language models capabilities, however these expect interfaces with specific formats, making integration unreliable...
     
Back in early 2023, I was trying to call an API to answer queries in natural language. APIs typically expect formatted inputs like JSON. Thus I was using a large language model to parse the queries.     
I was focusing on open source models, which were lagging behind GPT-3.5 at the time, let alone GPT-4, which was just released.     
Basically we had to answer two questions:
- Can a largue language model reliably parse natural language into a JSON? *nd: is it using valid keys and values*
- Is the resulting JSON accurate ? *nd: are the field values correct given the query*
     
The only way we knew how to make a LLM output a JSON was to provide instructions (with a great deal of uppercase and *IMPORTANT* blocks) including the JSON format and domain rules. We had to implement a few guardrails and parsing rules but we got a valid JSON most of the time. Was it reliable? Enough for the PoC. Just stray a little from the happy path and the model would come up with new keys, values, or would simply not be able to produce a valid JSON. GPT-3.5, which was our baseline, would also fall short in complex cases.

## Structured generation

I kept thinking about this, until I stumbled upon *structured generation* with [outlines](https://github.com/dottxt-ai/outlines).    
It does exactly what I was looking for: guarantee that a model's output would match a structure. 
      
Conceptually, there are two main approaches to structured generation. The first is based on validation and retries. The output is evaluated using a framework like Pydantic, and if it does not match, the model is asked to try again. This does not provide strong guarantee but it is often good enough in practice when using strong models. The second is *constrained generation*, forcing the model to output only specific tokens that match the structure. This is outline's approach. It is explained in details in their blog [Coalescence: making LLM inference 5x faster](https://blog.dottxt.ai/coalescence.html).    
      
Today, structured generation is everywhere, as it is a key feature of AI agents that need to interact with external systems. OpenAI has released it in their API and it is possible to provide a JSON to the `/v1/chat/completions` endpoint. Other tools that expose this endpoint like vLLM, Ollama, LM Studio... also implement it. However, LM Studio points that LLM below 7B parameters may not support it.
      

## Experiments

My point is to compare how structured generation improves a model's abilities for my JSON generation use case. I focused on rather small models that may be hosted on a single GPU or even locally, however I included a proprietary model (`gpt-4o-mini`) as a baseline.       
I used outlines and instructor to implement structured generation.    
I run the models locally using LM Studio's server. Everything runs on a Macbook Air M5 with 10 CPU cores.

I compared four models:
- `gpt-4o-mini` using OpenAI API (because it's cheap)
- `gemma-4-e2b-it` (4bit quantization)
- `qwen3-4b-instruct-2507-mlx` (4bit quantization)
- `qwen3-0.6b` (4bit quantization)

I compared the raw model generation to the structured generation with grammar-constraint (using outlines) and validate-retry (using [instructor](https://github.com/567-labs/instructor)).      
The use case consists in taking a candidate's description in natural language, and generate a JSON following this pattern:
```python
class ContactType(str, Enum):
    email = "email"
    phone = "phone"
    linkedin = "linkedin"
    github = "github"
    website = "website"


class Contact(BaseModel):
    type: ContactType
    value: str


class Occupation(str, Enum):
    software_engineer = "software_engineer"
    data_scientist = "data_scientist"
    data_engineer = "data_engineer"
    ml_engineer = "ml_engineer"
    devops_engineer = "devops_engineer"
    designer = "designer"
    product_manager = "product_manager"
    engineering_manager = "engineering_manager"
    researcher = "researcher"
    other = "other"


class Person(BaseModel):
    first_name: Optional[str] = Field(description="First/given name")
    last_name: Optional[str] = Field(description="Last/family name if mentioned")
    age: Optional[int] = Field(description="Age in years if mentioned")
    occupation: Optional[Occupation] = Field(description="Closest matching occupation category")
    contacts: list[Contact] = Field(description="Contact details found in the text")
```
I used Opus 4.6 to generate descriptions and their matching JSON ground-truth.     
You may find the code in this [github repository]().    
    
#### Simple case

To get started and iterate fast, I only used three descriptions at first ([`data/sample.csv`]() in the repo).      
I computed three main metrics:
- **Parsing rate**: how often the model outputs a valid JSON?
- **Accuracy**: how often the model outputs the correct fields? (based on the ground-truth).   
*nd: one could be interested in a per-key metric, but it was not the point here*
- **Average latency**: how long the model takes to output the JSON?
       
| Label | Valid | Invalid | Total | Parse rate | Matches | Mismatches | Total with ground truth | Accuracy | Avg duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai / gpt-4o-mini / raw | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 2.916 |
| openai / gpt-4o-mini / outlines | 3 | 0 | 3 | 1.0 | 2 | 1 | 3 | 0.666 | 3.022 |
| openai / gpt-4o-mini / instructor | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 1.901 |
| local / gemma-4-e2b-it / raw | 3 | 0 | 3 | 1.0 | 2 | 1 | 3 | 0.666 | 1.709 |
| local / gemma-4-e2b-it / outlines | 3 | 0 | 3 | 1.0 | 2 | 1 | 3 | 0.666 | 1.871 |
| local / gemma-4-e2b-it / instructor | 3 | 0 | 3 | 1.0 | 2 | 1 | 3 | 0.666 | 1.794 |
| local / qwen3-4b-instruct-2507-mlx / raw | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 2.090 |
| local / qwen3-4b-instruct-2507-mlx / outlines | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 1.836 |
| local / qwen3-4b-instruct-2507-mlx / instructor | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 1.856 |
| local / qwen3-0.6b / raw / thinking:on | 1 | 2 | 3 | 0.333 | 1 | 0 | 1 | 1.0 | 7.691 |
| local / qwen3-0.6b / raw / thinking:off | 0 | 3 | 3 | 0.0 | 0 | 0 | 0 | 0.0 | 1.064 |
| local / qwen3-0.6b / outlines | 3 | 0 | 3 | 1.0 | 0 | 3 | 3 | 0.0 | 0.804 |
| local / qwen3-0.6b / instructor | 3 | 0 | 3 | 1.0 | 0 | 3 | 3 | 0.0 | 0.907 |

This sample is very simple, so the best models get it right without structured generation. Only Qwen3-0.6B is struggling. This is especially interesting as it is by far the smallest model, and the one that benefits most from structured generation.    
Qwen3-0.6B has thinking enabled by default, I also compared its performance without thinking to get a better idea of the latency. Indeed, structured generation prevents thinking tokens to be emitted as these are not part of the grammar, hence cancelling the reasoning capabilities of this model. This is why the latency using structured generation is much lower. It should be compared with the `thinking:off` latency.      
Structured generation seems to make Qwen family models faster, however this is unconclusive from this experiment. It greatly improves the parsing ability of the smallest model, going from **0%** with thinking:off, to a perfect **100%**. Structured generation does not seem to have an impact on accuracy, but given the small sample size, it is yet unconclusive at this stage.

#### Tricky case

Three descriptions was not enough to push the models to their limits and I needed something more complicated, so I generated 35 new descriptions using Opus 4.6 and asked for tricky cases with misleading information.      
     
| Label | Valid | Invalid | Total | Parse rate | Matches | Mismatches | Total with ground truth | Accuracy | Avg duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai / gpt-4o-mini / raw | 35 | 0 | 35 | 1.0 | 6 | 29 | 35 | 0.171 | 2.688 |
| openai / gpt-4o-mini / outlines | 35 | 0 | 35 | 1.0 | 5 | 30 | 35 | 0.143 | 2.803 |
| openai / gpt-4o-mini / instructor | 35 | 0 | 35 | 1.0 | 8 | 27 | 35 | 0.229 | 1.821 |
| local / gemma-4-e2b-it / raw | 34 | 1 | 35 | 0.971 | 21 | 13 | 34 | 0.618 | 2.261 |
| local / gemma-4-e2b-it / outlines | 35 | 0 | 35 | 1.0 | 21 | 14 | 35 | 0.6 | 3.022 |
| local / gemma-4-e2b-it / instructor | 35 | 0 | 35 | 1.0 | 21 | 14 | 35 | 0.6 | 3.006 |
| local / qwen3-4b-instruct-2507-mlx / raw | 33 | 2 | 35 | 0.943 | 21 | 12 | 33 | 0.636 | 3.663 |
| local / qwen3-4b-instruct-2507-mlx / outlines | 35 | 0 | 35 | 1.0 | 22 | 13 | 35 | 0.629 | 2.923 |
| local / qwen3-4b-instruct-2507-mlx / instructor | 35 | 0 | 35 | 1.0 | 21 | 14 | 35 | 0.6 | 3.121 |
| local / qwen3-0.6b / raw / thinking:on | 14 | 21 | 35 | 0.4 | 3 | 11 | 14 | 0.214 | 9.554 |
| local / qwen3-0.6b / raw / thinking:off | 0 | 35 | 35 | 0.0 | 0 | 0 | 0 | 0.0 | 1.476 |
| local / qwen3-0.6b / outlines | 35 | 0 | 35 | 1.0 | 9 | 26 | 35 | 0.257 | 0.804 |
| local / qwen3-0.6b / instructor | 35 | 0 | 35 | 1.0 | 5 | 30 | 35 | 0.143 | 0.827 |


These results demonstrate the impact of structured generation more clearly. Although `gpt-4o-mini` can still parse without any help but prompting, other models start to show their limits. Although larger open source models perform well without structured generation (97% for gemma-4 and 94% for Qwen3-4B), these would still pose issues at scale.      
Structured generation makes all outputs valid JSON. As outlines enforces strict grammar this is not surprising. On the other hand, Instructor *only* does prompting, validation, and retries, yet it enables even the smallest model to reach perfect a parsing rate on this dataset. From this I conclude that although Instructor's approach does not provide theoretical guarantees, it is robust in practice.   
Does that mean that structured generation makes all these models useable in production? Not yet. The accuracy remains pretty low and neither framework seems to improve it. This is not surprising as outlines enforces grammar but does not improve reasoning abilities or semantic understanding. Same for Instructor which validates format, but does not affect reasoning.   
Does structured generation make inference faster? In the case of outlines, coalescence should indeed speed up decoding. Again from this experiment, this seems to depend on the model. Qwen models latency is indeed reduced on average, significantly for Qwen3-0.6B (~45% faster). However it does not seems to make a difference for `gemma-4` and `gpt-4o-mini`.

## Limitations

My main goal was to check if structured generation guaranteed reliable output structure. This is a success.       
What structured generation does not do, is make the models smarter. Small models may produce a valid output given the structure, but still pick the wrong value. This depends on the model's reasoning abilities. `outlines` only enforces the grammar, and `instructor` only improves prompting and adds a retry loop.           
Improving the semantic understanding of a model on this use case would require fine tuning in addition to structured generation.    
Another important practical point is the package sizes. I have tried to use `outlines` in a Lambda function and unfortunately it made the function artifact too large. On my machine, `outlines` requires 12M of disk space, and `instructor` 7.6M. This may prevent these tools to be used on devices where memory is tight.     

    
## Conclusion

Proprietary models are really good at parsing valid outputs based on instructions alone. For more complex cases, providers have a structured output API to enforce compliance to a schema. Hence for these models I would say that using a library for structured generation is not necessary.        
Regarding `outlines` my conclusion is that it should be used when one needs theoretical guarantees that the output will respect a format. Otherwise, for open source models, `instructor` is a good fit.   
Small open source models benefit the most from structured generation. Although it makes these output valid format, it does not improve the models cognitive abilities. These tools can be used to structure data in relatively simple cases, where constraints like privacy and cost would be priorities. However, these would require fine-tuning for more complex cases to extract accurate data.      

