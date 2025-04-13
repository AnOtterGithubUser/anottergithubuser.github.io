---
title: "What happens behind `vllm serve`"
date: 2025-03-24 11:22 +1100
excerpt: 
category: dive llm generativeai
---
vLLM is what first sparked my interest in LLM deployment -- although it was not the first framework I used. I actually started out with TGI, running in a dedicated container on AWS Sagemaker. vLLM, however, quickly became my go-to solution for serving LLM, mainly because of its performance and ease of use.    
      
Iin a previous [peek]({{ site.url }}/peeks/llm-inference-engines-servers/), I covered the basics of modern LLM serving frameworks. At their core are the concepts of LLM engine and LLM server. That earlier post just scratched the surface. In this one, I will take a much deeper dive and we will see how these are actually implemented in vLLM.      
      
## How to use vLLM's engine?
       
I previously discussed how vLLM's main optimizations come from its engine. It can be instantiated and used offline on its own.    
     
```python
from vLLM import LLM

llm = LLM(model="mistralai/Mistral-Small-3.1-24B-Instruct-2503") # path to the model repository in Hugging Face Hub
```
<p align="center">
  <em style="font-size: 0.8em;">Initialize a vLLM engine for Mistral Small 3.1</em>
</p>
     
This is not how I have been using vLLM in practice. For a full serving experience, you can run the engine within an inference server that mimics the OpenAI API protocol.
     
```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503
```
<p align="center">
  <em style="font-size: 0.8em;">Serves a vLLM engine for Mistral Small 3.1 with OpenAI API protocol</em>
</p>
     
This is how I have been using vLLM. This command spins up a FastAPI server that makes the engine available for online inference. As you can see, the command is very straightforward -- just run it on a capable machine (ideally with a GPU) and you are good to go (at least for prototyping).   
This simplicity masks the optimizations implemented under the hood. You might be using vLLM without even realizing it -- like I was, as I explained [there]({{ site.url }}/peeks/llm-inference-engines-servers/). So, what is really going on behind `vLLM serve`?
       
## Command line entrypoints
     
*From this point on, I will refer to the [vLLM github repository](https://github.com/vllm-project/vllm/tree/main) as "the repository" or "vLLM repository".*     
    
Where to start? At the root of the repository is a `pyproject.toml` file. It acts as the central configuration for the project. It defines things like dependencies, settings, and more. The line we care about is this one.
     
```yaml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```
     
This tells us where the command lines for vLLM are defined. The one we want is in `vllm/entrypoints/cli/serve.py`.    

<details class="note-toggle">
  <summary><em>Note</em></summary>
  <div>
    <p>
        <em>vLLM defines its command line interface with <a href="https://docs.python.org/3/library/argparse.html">argparse</a>, a popular choice as Python's built-in library. Although it is widely used, I personally lean toward <a href="https://github.com/pallets/click">click</a> in my own projects. It is a modern library whose composability and decorator-based syntax make command line definitions more readable -- especially complex ones
        </em>
    </p>
  </div>
</details>
<p></p>
Here is the command that is called to start the server.
     
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
     
`uvloop` is a Python library for running high performance event loops for asyncio. Here, it runs the async function `run_server` indefinitely -- until the process is manually stopped.  
     
## vLLM inference server
     
The `run_server` function will launch vLLM's inference server, built with FastAPI.
The FastAPI application is initialized with this line.   
```python
app = build_app(args)
```
Then, the application state is populated with vLLM-specific variables:
```python
await init_app_state(engine_client, model_config, app.state, args)
```
In FastAPI, the application state is a collection of objects that need to persist throughout the application's lifetime -- like the engine, logs, metrics, and more.
The application is then bound to an `APIRouter`, a `FastAPI` object that defines the routes available on the server. The route we are interested in is `/v1/chat/completions`. As the name suggests, it is used for chatting with the model.      
Since it follows the same format as OpenAI API, a model served with vLLM can be used as a drop-in replacement for OpenAI models, with minimal changes.    
For example, in a LangChain application, one would normally use the `ChatOpenAI` class to use OpenAI models. To switch to a vLLM-served model, only the `base_url` argument shall be updated to your vLLM server's URL.
     
