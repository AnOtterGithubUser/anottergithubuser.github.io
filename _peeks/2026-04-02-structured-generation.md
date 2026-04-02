---
title: "Structured outputs from large language models"
date: 2026-04-02 15:13 +100
excerpt: How to make a LLM generate structured output with great reliability
category: peek generativeai coding llm
---

The large language models return a sequence from a sequence, that's a simple way to say it. In its most popular form, these sequences are text (chatGPT) and this enables LLM to hold a conversation. Since they are non-deterministic (i.e do not always yield the same result for the same input), it feels natural. Although it's great for an everyday chatbot to mimic intelligence, it prevents these models to be integrated with other pieces of software, because these expects clear structure.

## The problem with non-determinism

When you keep asking the same question to a LLM, the answer will be slightly different each time, although the general intention will remain the same. This is a feature, not a bug. These models craft their answers by sampling from words in their vocabulary (*nd: LLM sample from tokens, not word, but in this article the distinction does not matter*). The more relevant a word is in the context, the more likely it is to be sampled. However, there is no guarantee that a specific word will be picked.      
This becomes an issue when we want to integrate a LLM with other deterministic systems which expect reliability and specific inputs.       
Back in spring 2023, generative AI pre-history, I was working with open source models. There was not much at this time, open source was lagging behind GPT-3.5, let alone GPT-4 which had just been released. Basically, I needed to build a search engine based on natural language, that would look into a database and return the most relevant items. Imagine asking a sport retailer AI for a pair of running shoes and it would return 4-5.       
Today, any engineer would say "that's just a function call to use a tool, any AI agent gets that now". True. We were actually building a specialized agent that would look into the retailers database based on a query. However these terms did not exist yet in early 2023.      
To use, it meant calling an API, which is a deterministic system, from a query. The API does not understand natural language ("I need a pair of running shoes"), so the LLM comes in between to turn the query into a structured format, which is usually JSON. There were two main questions:
- Is the LLM able to make a JSON out of a query ?
- How reliable is it at doing so ?       
     
LLM are able to follow instructions, so the only way we knew how to output a JSON was simply to tell it to do so (with a great deal of uppercase and *IMPORTANT* blocks). So, to answer the first question, did it work? Well, kinda. We had to implement a few guardrails to make sure we could parse the output but we would get a correct JSON more times than not.      
Was it reliable? Absolutely not. Ask something a little outside of the happy path and it would come up with fields that did not exist in the API, badly formatted values, or even could not produce a reliable JSON. This is not something you can put in production, not when the point was for customers to find items in the catalog. GPT-3.5 was our baseline and although it was performing better, it would still fail when the query was too hard.       

## Structured generation

Was there a way to guarantee that the model would output a JSON? No.     
Forget about parsed outputs in the OpenAI API, it did not exist yet, and our point was to use open source models anyway. That also meant self hosting a model, which meant GPU, which meant good money, so the smaller the better. But the smaller the model was, the worse it would perform.       
We managed to deliver a v1 with a huge set of instructions and clever output parsing. That was enough for the PoC, but it always bugged me thinking back.     
Until a year ago, when I stumbled upon [outlines](https://github.com/dottxt-ai/outlines). I have been wanting to write about it since, but only found the time now.      
Outlines does exactly what I was looking for: guarantee that the output of a model would follow a certain structure. It does it by turning the expected structure into a regular expression, and then force the model to output only tokens that match this expression at each decoding step.      
When a LLM generates a new token, it samples from its entire vocabulary. This is very inefficient to generate a JSON as only a small subset of the vocabulary matches the regular expression. Outlines sets the probability of tokens outside of this subset to zero, thus preventing the model to pick them with 100% reliability. It also speeds up the generation as the LLM does not need to be called to get new tokens when there is only one way to match the regular expression. They call this coalescence and it is explained in greater details in their [blog article](https://blog.dottxt.ai/coalescence.html).        
In addition to all that, their approach is agnostic to the model!       
      
We are in 2026 now, and OpenAI released structured generation in their API through the `/v1/chat/completions` endpoint. They do accept a JSON schema as a response format. Servers like vLLM, LM Studio, Llama.cpp, Ollama... use the same endpoints and may work with this structured output API depending on the model. However, LM studio points that LLMs below 7B parameters may not support it.       

## Experiment

I ran a simple experiment to assess Outline's capabilities. I compared four models:
- `gpt-4o-mini` using OpenAI API (because it's cheap)
- `mistralai_-_mistral-7b-instruct-v0.3` using LM Studio
- `qwen3-4b-instruct-2507-mlx` using LM Studio
- `qwen3-0.6b` using LM Studio
I compared the raw model generation (without structured output constraint) to the structured generation (using outlines).      
The use cas was to take a candidate's description in natural language, and generate a JSON following this pattern:
```json
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
I used Opus 4.6 in Claude Code to generate descriptions and their matching JSON ground-truth.     
You may find the code in this [github repository]().    
    
#### Simple case

To get started and iterate fast, I only used three descriptions at first (`data/sample.csv` in the repo):
```
"Alice is a 32-year-old machine learning engineer at Acme Corp. She specializes in Python, PyTorch, and distributed training. Reach her at alice@acme.com or on LinkedIn at linkedin.com/in/alice.","{""first_name"": ""Alice"", ""last_name"": null, ""age"": 32, ""occupation"": ""ml_engineer"", ""contacts"": [{""type"": ""email"", ""value"": ""alice@acme.com""}, {""type"": ""linkedin"", ""value"": ""linkedin.com/in/alice""}]}"
"Bob, 45, is a freelance data scientist and former academic. His work spans R, SQL, and causal inference. Contact: bob@datascience.io, GitHub: github.com/bsmith.","{""first_name"": ""Bob"", ""last_name"": null, ""age"": 45, ""occupation"": ""data_scientist"", ""contacts"": [{""type"": ""email"", ""value"": ""bob@datascience.io""}, {""type"": ""github"", ""value"": ""github.com/bsmith""}]}"
"Carol runs her own UX design studio. She's been in the field for 12 years and is proficient in Figma, user research, and prototyping. Website: caroldesigns.com.","{""first_name"": ""Carol"", ""last_name"": null, ""age"": null, ""occupation"": ""designer"", ""contacts"": [{""type"": ""website"", ""value"": ""caroldesigns.com""}]}"
```
I computed three main metrics: the rate of correctly formatted JSON, the average model latency, and the accuracy of the output JSON. The accuracy here is very simple, 1 if the output matches the ground truth, 0 otherwise. One could compute a per-field metric but this was not the point here.     
       
| Provider | Model | Outlines | Valid | Invalid | Total | Parse rate | Matches | Mismatches | Total w/ ground truth | Accuracy | Avg duration (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai | `gpt-4o-mini` | NO | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 2.4152286943329577 |
| openai | `gpt-4o-mini` | YES | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 2.8009182780006086 |
| local | `mistralai_-_mistral-7b-instruct-v0.3` | NO | 3 | 0 | 3 | 1.0 | 2 | 1 | 3 | 0.6666666666666666 | 4.086129889333582 |
| local | `mistralai_-_mistral-7b-instruct-v0.3` | YES | 3 | 0 | 3 | 1.0 | 2 | 1 | 3 | 0.6666666666666666 | 4.092326889333587 |
| local | `qwen3-4b-instruct-2507-mlx` | NO | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 1.8482286940003785 |
| local | `qwen3-4b-instruct-2507-mlx` | YES | 3 | 0 | 3 | 1.0 | 3 | 0 | 3 | 1.0 | 1.7612535140009034 |
| local | `qwen3-0.6b` | NO | 3 | 0 | 3 | 1.0 | 1 | 2 | 3 | 0.3333333333333333 | 7.367193555333263 |
| local | `qwen3-0.6b` | YES | 3 | 0 | 3 | 1.0 | 0 | 3 | 3 | 0.0 | 0.5796493613318793 |
       
*nd: on a macbook air M5*
      
The sample is quite simple so unsurprisingly, `gpt-4o-mini` performs perfectly with or without outlines. All models output a valid JSON 100% of the time. The smartest open source model, Qwen3 4B also has a perfect accuracy. However using outlines does not seem to improve the accuracy, although the model uses valid keys and values, it may still pick the wrong one. For example, in the third description, Mistral 7B and Qwen 0.6B output `"age": 12`.

#### Tricky case

Three descriptions was not enough to push the models to their limits and I needed something more complicated, so I generated 50 descriptions using Opus 4.6 and asked for tricky cases with misleading information.      
     
| Provider | Model | Format | Valid | Invalid | Total | Parse rate | Matches | Mismatches | Total w/ ground truth | Accuracy | Avg duration (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai | `gpt-4o-mini` | raw | 50 | 0 | 50 | 1.0 | 7 | 28 | 35 | 0.2 | 2.0838135257999237 |
| openai | `gpt-4o-mini` | outlines | 50 | 0 | 50 | 1.0 | 7 | 28 | 35 | 0.2 | 2.068227518340209 |
| local | `mistralai_-_mistral-7b-instruct-v0.3` | raw | 26 | 24 | 50 | 0.52 | 7 | 10 | 17 | 0.4117647058823529 | 10.899819450000287 |
| local | `mistralai_-_mistral-7b-instruct-v0.3` | outlines | 50 | 0 | 50 | 1.0 | 13 | 22 | 35 | 0.37142857142857144 | 6.841599299959853 |
| local | `qwen3-4b-instruct-2507-mlx` | raw | 47 | 3 | 50 | 0.94 | 20 | 13 | 33 | 0.6060606060606061 | 2.978806803339976 |
| local | `qwen3-4b-instruct-2507-mlx` | outlines | 50 | 0 | 50 | 1.0 | 21 | 14 | 35 | 0.6 | 2.4214638374601782 |
| local | `qwen3-0.6b` | raw | 21 | 29 | 50 | 0.42 | 6 | 11 | 17 | 0.35294117647058826 | 11.35391688082018 |
| local | `qwen3-0.6b` | outlines | 50 | 0 | 50 | 1.0 | 6 | 29 | 35 | 0.17142857142857143 | 0.7671098617999087 |
       
Here the results are much more interesting and show the impact of outlines. Although `gpt-4o-mini` can still output valid JSONs without the help of outlines, it is not the case of smaller open source models. Qwen3-0.6B and Mistral 7B fall to ~50% successfull formatting. This would not be usable in production. Even the ~94% of Qwen3-4B would pose significant issues at scale. However, outlines make all parsing reliable. Does it mean these models can be used in production? Not yet. Indeed the accuracy is too low, even for `gpt-4o-mini`.      
I also wanted to check if Outline indeed speeds up LLM generation. From the results it seems to be the case, although it is not conclusive. It looks like it really depend on the model and I did not get such latencies at each run.

## Limitations

My main goal was to check if Outline's structured generation guaranteed reliable output structure. This is a success. It is not a trial and error loop like the first structured generations frameworks, but a real mathematical constraint enforcing structure.     
What Outline does not do, is make the LLM smarter. The model may produce a valid output given the structure, but it may still pick the wrong value. This depends on the model's reasoning capabilities, Outline only enforces the grammar.       
Hence these models would still require fine-tuning.

    
## Conclusion

Proprietary models are already really good at generating structured outputs based on instructions alone, and for more complex cases, the provider has a structured output API. The point of Outline is in small open source models. Indeed one does not need the full reasoning capabilities of Opus 4.6, GPT-5.4, or Gemini 3, to simply perform sentiment analysis or output JSON. Often, very small models, even under 1B like we saw, are capable to perform well with Outlines.
