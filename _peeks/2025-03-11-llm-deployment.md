---
title: "LLM deployment: an early 2025 review"
date: 2025-03-11 15:47 +1100
excerpt: Interest in Large Language Models is growing fast, so have their abilities in the last 2 years, but how can you use them in your organization?
category: peek llm generativeai
---
A year ago I wrote an article while I was at OCTO Technology where I presented the [different ways one could deploy an open source LLM](https://blog.octo.com/comment-utiliser-un-llm-open-source-1). This is a particularly interesting subject to me, so I kept digging.     
       
A lot has changed in just two years in the open source landscape of generative AI. I started working with open source LLMs in spring 2023. Privacy and ownership were the main motivators but back then the offer was slim. Basically you could choose between Llama and Llama.      
Jokes aside, there were a few more. I experimented with T5 (Google), Dolly (Databricks), Red Pajama (Stanford), etc. However, GPT4 was just released and these were far from its level, and to be honest, not even on GPT 3.5 level. Not to belittle the teams who worked on these models. LLMs were a new thing, GPT 3.5 was barely six months old, and as we saw later, it was just a matter of time for open source to catch up.     
Anyway, in April 2023, we built a langchain application for our use case. We were working on AWS so we deployed the models on Sagemaker endpoints. And then another issue arose...      
      
It was slow! So painfully slow, even for relatively small models. Not suitable for a customer facing solution. We used the pytorch hugging face deep learning inference image on Sagemaker and obviously it was not suited for this job. We were far from the speed of ChatGPT and its token streaming capabilities.    
Fortunately, Hugging Face released their [TGI powered inference container on Sagemaker](https://huggingface.co/blog/sagemaker-huggingface-llm), which gave our application a significant speed up!
     
      
### LLM deployment options
     
As of my previous article at OCTO, the main tools remain.    
For cloud based:
- vLLM
- TGI
- TensorRT + Triton
For local deployment:
- Llama.cpp
- Ollama
With some additions though

