---
title: "What happens behind `vllm serve`"
date: 2025-03-24 11:22 +1100
excerpt: 
category: dive llm generativeai
---
vLLM is what got me interested in LLM deployment optimizations, though it was not the first framework I tried. I actually used TGI first, with its dedicated container on AWS Sagemaker. vLLM, however, quickly became my go-to solution for serving LLM, mostly because of its performances and easiness to use.    
      
I introduced the basics of modern LLM serving frameworks in a previous [peek]({{ site.url }}/peeks/llm-inference-engines-servers/). At the core of their optimizations is the concept of LLM engine and LLM server. The peek however, just scratched the surface. In this post, I will dig much deeper and we will see how exactly these are implemented in vLLM.      
      
## How to use vLLM's engine?
       
I previously discussed how vLLM's main optimization come from its engine. It can be instantiated and used on its own offline.    
     
```python
from vLLM import LLM

llm = LLM(model="mistralai/Mistral-Small-3.1-24B-Instruct-2503") # path to the model repository in Hugging Face Hub
```
<p align="center">
  <em style="font-size: 0.8em;">Initialize a vLLM engine for Mistral Small 3.1</em>
</p>
     
This is not how I have been using vLLM though. For a full serving experience, you can serve the engine with an inference server that reproduces OpenAI API protocol.
     
```bash
>> vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503
```
<p align="center">
  <em style="font-size: 0.8em;">Serves a vLLM engine for Mistral Small 3.1 with OpenAI API protocol</em>
</p>
     
THIS is how I have been using vLLM. It will start a FastAPI server to use the engine for online inference. As you can see, it is very simple. Run this command on a capable machine (preferably with a GPU) and you are good to go (for prototyping at least). This simplicity hides the optimizations of vLLM, and you may be using it without a clue of what is happening behind (that was my case as I described [there]({{ site.url }}/peeks/llm-inference-engines-servers/)). Now what is really happening?
       
## The entrypoint: `vLLM serve`
     
From this point onward, I will refer to the [vLLM github repository](https://github.com/vllm-project/vllm/tree/main) as "the repository" or "vLLM repository".     
    
Where to start? At the root of the repository, you will find a `pyproject.toml`. This file centralizes the configuration of the project. In it, there are things like the dependencies, settings, etc. Our main interest is this line.
     
```yaml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```
     
It is pointing us to the command lines that are included in vLLM. The one we want is here `vllm/entrypoints/cli/serve.py`. We notice that vLLM uses the `argparse` library.    
This what starts the server:
     
```python
class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "serve"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        uvloop.run(run_server(args))
```
     
`uvloop` is a Python library for running asyncio event loops. Here, it will run an async function indefinitely until stopped. This async function is `run_server`.
     
## vLLM inference server
     
`run_server` will launch the inference server of vLLM which is a FastAPI app.    
The application is created here.   
```python
app = build_app(args)
```
It will then populate the application state with vLLM variables.
```python
await init_app_state(engine_client, model_config, app.state, args)
```
In FastAPI, the application state is a collection of objects which persists during the application's lifetime.   
The application is bound to an `APIRouter` which is a `FastAPI` object. It defines the routes that will be reachable on the server. The one we are interested in is `/v1/chat/completions`. As its name suggests, it is used to chat with the model. Since it is the same as OpenAI API, the model we serve can be used as a drop-in replacement for OpenAI models, with minimal efforts.    
For example, in a LangChain application, one would use the `ChatOpenAI` class to integrate an OpenAI model. The only thing that would need to change to switch to a vLLM served model would be the `base_url` argument. It should be the url of the server where vLLM is running.
     
