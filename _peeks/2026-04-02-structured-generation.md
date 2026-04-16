---
title: "Improve small models reliability with structured generation"
date: 2026-04-02 15:13 +100
excerpt: Using structured generation to make LLM reliable
category: peek generativeai coding llm
---

A few years ago, when GPT-4 had just been released, I had the opportunity to work on a project involving small open source LLMs. Good models were scarce and I did not have the infrastructure to run the largest ones (Llama 65B back then). The problem was to take a query in natural language as input, and call an API to fetch the required data. This usually involves turning text into a structured format like JSON.     

# Structured generation with LLMs

You probably have heard about structured generation for some times, if not, you sure have benefited from it without knowing.     
Language models can follow instructions to perform various tasks, which makes them exceptionnaly useful and versatile. However, you may have noticed they always answer the same question in a slighlty different way. This makes interactions feel more natural, but it is an issue when interacting with external systems which expect specific formats. Back in early 2023, we had to use a large prompt, few-shot learning, and clever parsing to make sure the output was a valid JSON.    
That is what **Structured Generation** is for: make sure a model always answers using the same structure. AI agents rely on it extensively today to call external APIs and interact with the outside world.     
OpenAI introduced their [Structured Output API](https://openai.com/index/introducing-structured-outputs-in-the-api/) in summer 2024. It is based on a technic called **constrained-decoding**. Basically, language models work by sampling tokens from their vocabulary, which we call the *support*. The more relevant a token is, the more likely it is to be sampled. Constrained decoding works by restricting the support to a subset of tokens compliant with the provided format. Outlines, an open source constrained decoding library, explains it in details in their blog post [Coalescence: making LLM inference 5x faster](https://blog.dottxt.ai/coalescence.html).      

# Where constrained decoding happens

To restrict the support, i.e the tokens the model can sample from, constrained decoding applies a mask. This mask sets the probability of non-valid tokens to zero.     
For example, in order to generate a valid JSON, the first token must be `{` (or start with `{ `), so the probability of all the other tokens is set to zero, hence the decoder has no choice but pick `{`.   
This means that constrained decoding need to access the *probabilities* of each token. So it must live in the same process as the decoder. That means that constrained decoding usually lives in the **server** with the model. Actually it always does in production.    
Open source inference engines like vLLM or TGI enable constrained decoding using the same API as OpenAI. To do so, they leverage tools like xgrammar, LLGuidance ([vLLM](https://docs.vllm.ai/en/latest/features/structured_outputs/)), or outlines ([TGI](https://docs.vllm.ai/en/latest/features/structured_outputs/)). However these engines are designed to work with GPUs, and I don't have one...    
Luckily, local engines like llama.cpp, Ollama, or LM Studio, also implement this feature.      
Now, although it's interesting to know how this works, you will likely **never** use a library like outlines in your application. You will only use it (or an equivalent) indirectly through the structured output API and the `response_format` argument.    
So **why** am I telling you this? Because it personnally took me some time to understand it. Also because outlines does not tell you straight away, it actually falls back to the structured output API when you call a server.      

# Small language model for document understanding

Document understanding involves extracting text from a picture, and then structuring it. It is usually applied on forms where values must be matched to their respective keys. Usually, it requires chaining two models, one to extract the text (like Tesseract), and then another to perform matching (a LLM, or just a regex sometimes). Visual Language Models enable to do it in one pass.    

I ran an experiment using a VLM locally in LM Studio. I used Qwen3-VL-2B-Instruct, which offers a good trade-off between speed, performance, and memory.    
The model must generate a JSON from the image of an invoice.    
<ADD PYDANTUC MODEL FOR INVOICE>     
     
## Experiment baseline: Prompting

As a baseline, I started with simply prompting the model, including the JSON schema in the prompt. Out of 3 invoices, the model never generates a valid JSON. It include markdown format, escape characters, ... Although it understood the task and can mostly read the data, it fails at following the instructions precisely regarding the format. Up to mid-2024, we would have implemented guardrails, parsed the output, added instructions, but we would never have had any guarantee. Fortunately it is now 2026.     

## Improvement: Constrained decoding

We now add the JSON schema we expect in the `response_format` argument. Since the model uses MLX as inference engine, LM Studio uses outlines to implement constrained decoding. We now get a 100% valid outputs from the model.     
Is that it? Is OCR solved?    
No, check the values, these are not right.    
<ADD IMAGE AND OUTPUTS>  

# Structural vs Semantic validity

Structured generation guarantees that the output will follow a given structure. This means that it will use only existing keys, will not forget any, and that every value will have the right type. However, it does not guarantee that the values will be correct. For example, our model read `65.9` instead of `65.99`. The type is correct, the structure is valid, constrained decoding cannot detect that. It only guarantees **structure**, it does not guarantee relevance nor coherence, which we call **semantic**.          
And this was expected, solving this problem cannot be as simple as calling an API. OpenAI also pointed this out in the limitations in their article I mentioned above.     
To ensure semantic validity, we must implement **domain rules**. Because your domain depends on your application, this part must live in your application layer on the client side.     
So on top of constrained decoding, we apply domain knowledge and a retry mechanism. Fortunately, we don't have to do this ourselves.    

## Enforcing semantic validity with instructor

[Instructor](https://python.useinstructor.com/concepts/) provides a unified interface to create clients for any provider. It leverages the structured output API of the providers and implements a validation and retry mechanism on top of it. Since LM Studio supports the OpenAI API, we can use the OpenAI client with a custom api URL pointing to localhost.     
We first define model validators in each Pydantic model. These work based on the assumption that each field will have the right format. In our case, we check that the dates make sense, and the operations between the different quantities work.    
<ADD PYDANTIC CODE>     
On each generation, Instructor will automatically run these checks. It will include the validation errors, if any, in a new prompt, and send it back to the model to auto-correct itself. This comes at the cost of increased latency.      
We now catch mistakes, like VAT not being correctly computed, or amounts not adding up. However, the model cannot always successfully fix the output. Although we can catch issues and attempt to fix them, thus decreasing the number of errors, there is no way to guarantee correctness. Even when the model fixes its answer to make it semantically valid, there is a risk that it made up a number instead of checking the picture again. Usually, instructor will retry at least once before making up an answer, so invoices that did not work on first try should go to a separate stream for human validation. This is the most reasonable approach to date.     

## Conclusion

Structured generation provides strong theoretical guarantees, but it is not a substitute for the model's intelligence. The model's ability to reason and correctly understand the input is still key. Although the structural validity is now a solved problem, semantic validity remains an issue. We have seen how to implement guardrails and add retries, but for small models it can be enforced, not guaranteed. Even when checks pass and the output is semantically valid, we should always add a human in the loop for these use cases. These tools mostly help us sorting the cases that can be inserted automatically and the ones that need human validation.

