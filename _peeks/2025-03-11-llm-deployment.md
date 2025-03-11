---
title: "LLM deployment: engine and server"
date: 2025-03-11 15:47 +1100
excerpt: Deploying a LLM on your own infrastructure is getting common, but how does it actually work?
category: peek llm generativeai
---
A year ago I wrote an article while I was at OCTO Technology where I presented the [different ways one could deploy an open source LLM](https://blog.octo.com/comment-utiliser-un-llm-open-source-1). This is a particularly interesting topic to me, so I kept digging.     
       
A lot has changed in just two years within the open source landscape of generative AI. I first started working with open source LLMs in the spring of 2023. Privacy and ownership were my main motivations but ,at the time, the options were limited. Basically you could choose between Llama... and Llama.      
Jokes aside, there were a few others. I experimented with T5 (Google), Dolly (Databricks), Red Pajama (Stanford), etc. However, GPT4 had just been released, and open source alternatives were not on par with GPT 3.5, let alone its successor. Not to diminish the great work behind these models. LLMs were a new thing, GPT 3.5 was barely six months old, and as we later saw, it was only a matter of time before open-source caught up.     
Anyway, in April 2023, we built a langchain application for our use case. Since we were working on AWS, we deployed the models on Sagemaker endpoints. And then another issue emerged...      
      
It was **slow** -- painfully slow. Even with relatively small models. Not suitable for a customer-facing solution. We used the PyTorch Hugging Face deep learning inference container on Sagemaker and obviously it was not suited for this job. We were far from the speed of ChatGPT and its token streaming capabilities.    
Fortunately, Hugging Face released their [TGI powered inference container on Sagemaker](https://huggingface.co/blog/sagemaker-huggingface-llm) a month later, which gave our application a **significant** speed boost!     
I got interested in deploying LLMs efficiently.
       
      
## LLM deployment options
     
As I noted in my previous article at OCTO, the most popular deployment tools mostly remain the same.     
For local deployment:      
- [Llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Ollama](https://github.com/ollama/ollama)
- *[MLC-LLM](https://github.com/mlc-ai/mlc-llm), [OpenLLM](https://github.com/bentoml/OpenLLM), ...*
 
For cloud-based deployment:      
- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
      
*Some providers offer even faster alternatives today. [Groq](https://groq.com/) is a great example, or even more recently [Cerebra which powers Mistral's Le Chat](https://cerebras.ai/blog/mistral-le-chat) at an impressive ~1000 tokens/second!*    
*However, these solutions are closed-source and hardware-dependant, thus I will not discuss them in this article.*     
             
vLLM has become my go-to solution to run a LLM on GPU. It is both very simple to install and to use. It serves a model with fastAPI, outputing comprehensive logs, and it also makes token streaming easy. It also provides  an OpenAI-like protocol, making it easy to integrate in an application.        
      
So I have used vLLM since 2023, without much consideration for what was really running behind the scene. Until recently when I stumbled upon something:
> Wait, why does Triton Inference Server have a vLLM integration? Doesn't vLLM already serve model? What's the point of Triton then?      

To me this did not make sense. So I dug into it and found concepts, hidden in the simple `vllm serve` command, that I did not pay much attention to at first. These are inference **engine** and **server**.
     
      
## LLM Engine
        
I have to admit, this concept was a bit foreign to me.      
I thought you would use a framework such as PyTorch or TensorFlow to load a model and perform the matrices multiplications required in the transformers architecture. Like a boosted numpy on GPU, right? Well, not quite.   
This is actually what we did back in early 2023 with our slow PyTorch container, and as I said, it did not work very well.     
A LLM requires something more sophisticated, with optimizations suited to its auto-regressive nature.      
vLLM does not simply call `transformers.load(...)` and call it a day. It builds an optimized LLM engine.      
*Note: TGI and TensorRT-LLM also build optimized engines*       
       
What would these optimisations be? Well regular PyTorch would compute matrices multiplications sequentially and every new token requires an entire pass through the whole model. An engine would implement mechanisms such as batching, KV-cache, ...     
A simple analogy would be a car's engine. In this case, PyTorch would be your regular Twingo engine, and vLLM would be a V8 with a turbo.      
      
However, an engine alone will not get you very far. You may use it in your application but then it would load the whole model in your application. That is not very efficient. You would need to access your engine's outputs easily, without having to know exactly what is happening inside. We would also like our application to be independent of the LLM we use, which is not possible if we build our engine in a script.     
What we need is an **inference server**.       

## Inference server
    
Triton is an inference server. It has nothing to do with LLM. But it enables to interact easily with it.


*Let's keep things simple here, however if you want to dive into what a LLM engine and inference server actually do, you can go to [this article]()*

