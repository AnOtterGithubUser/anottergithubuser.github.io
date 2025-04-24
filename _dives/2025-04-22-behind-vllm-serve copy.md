---
title: "What happens behind `vllm serve`"
date: 2025-04-22 16:56 +1100
excerpt: "You may already be using vLLM, but do you really know what happens when you run it?" 
toc: true
category: dive llm generativeai
---
vLLM is what first sparked my interest in LLM deployment -- although it was not the first framework I used. I actually started out with TGI, running in a dedicated container on AWS Sagemaker. vLLM, however, quickly became my go-to solution for serving LLM, mainly because of its performance and ease of use.    
      
In a previous [peek]({{ site.url }}/peeks/llm-inference-engines-servers/), I covered the basics of modern LLM serving frameworks. At their core are the concepts of **LLM engine** and **LLM server**. That earlier post just scratched the surface. In this one, I will take a much deeper dive and we will see how these are actually implemented in vLLM.      
      
## How to use vLLM's engine?
       
I previously discussed how vLLM's main optimizations come from its engine. It can be instantiated and used offline on its own.    
     
```python
from vllm import LLM

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
This simplicity masks the optimizations implemented under the hood. You might be using vLLM without even realizing it -- like I was. So, what is really going on behind `vllm serve`?

<p>
<em>
This post is divided in two parts<br>    
- <strong>`vllm serve` walkthrough</strong> is a step-by-step walk through the code of vLLM from the command execution to the core components<br>
- <strong>Understanding vLLM optimizations</strong> is an in-depth explanation of the theory behind the optimizations of vLLM<br>
The parts are largely independant
</em>
</p>     
      
## `vllm serve` workflow   

### Starting the server
   
Let's start from the above command a user would run in its terminal:     
     
```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503
```
     
The command calls a script that relies on vllm CLI. This script is defined in [`vllm/pyproject.toml`](https://github.com/vllm-project/vllm/blob/main/pyproject.toml) at the root of the repository. This kind of file is used with popular project management tools like [Poetry]() to act as a central configuration for settings, dependencies, scripts, and more.    
     
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>[`vllm/pyproject.toml`](https://github.com/vllm-project/vllm/blob/main/pyproject.toml)</code></figcaption>
{% highlight yaml %}
41 [project.scripts]
42 vllm = "vllm.entrypoints.cli.main:main"
{% endhighlight %}
</figure>
     
The entrypoints is in [`vllm/vllm/entrypoints/cli/main.py`](vllm/vllm/entrypoints/cli/main.py) which is a dispatch for the subcommands of the command line (`serve`, `openai`, `benchmark.main`, `collect_env`). The user command would run the `serve` subcommand with the positional argument `'mistralai/Mistral-Small-3.1-24B-Instruct-2503'`.    
      
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>[`vllm/entrypoints/cli/serve.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/cli/serve.py)</code></figcaption>
{% highlight python %}
...
8  from vllm.entrypoints.openai.api_server import run_server
...
14 class ServeSubcommand(CLISubcommand):
15     """The `serve` subcommand for the vLLM CLI. """
16 
17     def __init__(self):
18         self.name = "serve"
19         super().__init__()
20 
21     @staticmethod
22     def cmd(args: argparse.Namespace) -> None:
23         # If model is specified in CLI (as positional arg), it takes precedence
24         if hasattr(args, 'model_tag') and args.model_tag is not None:
25             args.model = args.model_tag
26 
27         uvloop.run(run_server(args))
{% endhighlight %}
</figure>
      
It passes the argument to a `run_server` function which runs in an asyncio event loop. This function essentially builds a FastAPI application and injects the core components of vLLM in its state.    
       
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>[`vllm/entrypoints/openai/api_server.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py)</code></figcaption>
{% highlight python %}
1041 async def run_server(args, **uvicorn_kwargs) -> None:
...      
1077     async with build_async_engine_client(args) as engine_client:
1078         app = build_app(args)
1079
1080         vllm_config = await engine_client.get_vllm_config()
1081         await init_app_state(engine_client, vllm_config, app.state, args)
...
{% endhighlight %}   
</figure>   
     
The core logic is implement in the engine client. It is an asynchronous client provided in a context manager. Following the stack trace, it is initialized in the `build_async_engine_client` function, which essentially calls the `build_async_engine_client_from_engine_args` function of the same file.   

<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/entrypoints/openai/api_server.py</code></figcaption>
{% highlight python %}
138 @asynccontextmanager
139 async def build_async_engine_client(
140         args: Namespace) -> AsyncIterator[EngineClient]:
141 
142     # Context manager to handle engine_client lifecycle
143     # Ensures everything is shutdown and cleaned up on error/exit
144     engine_args = AsyncEngineArgs.from_cli_args(args)
145 
146     async with build_async_engine_client_from_engine_args(
147             engine_args, args.disable_frontend_multiprocessing) as engine:
148         yield engine
149 
150 
151 @asynccontextmanager
152 async def build_async_engine_client_from_engine_args(
153     engine_args: AsyncEngineArgs,
154     disable_frontend_multiprocessing: bool = False,
155 ) -> AsyncIterator[EngineClient]:
...
168     # V1 AsyncLLM.
169     if envs.VLLM_USE_V1:
...
177         try:
178             async_llm = AsyncLLM.from_vllm_config(
179                 vllm_config=vllm_config,
180                 usage_context=usage_context,
181                 disable_log_requests=engine_args.disable_log_requests,
182                 disable_log_stats=engine_args.disable_log_stats)
183             yield async_llm
... 
188     # V0 AsyncLLM.
189     elif (MQLLMEngineClient.is_unsupported_config(vllm_config)
190           or disable_frontend_multiprocessing):
191 
...
193         try:
194             engine_client = AsyncLLMEngine.from_vllm_config(
195                 vllm_config=vllm_config,
196                 usage_context=usage_context,
197                 disable_log_requests=engine_args.disable_log_requests,
198                 disable_log_stats=engine_args.disable_log_stats)
199             yield engine_client
{% endhighlight %}   
</figure>      

vLLM V1 released its alpha in January 2025 and introduces significant upgrades which are beyond the scope of this post. As of the date I am writing this, April 2025, we assume the user has already switched to vLLM V1. So the engine client here is an instance of `AsyncLLM`.    
`AsyncLLM` may take some time to initialize, the server will start as soon as it is ready. Let's dig in its initialization process to understand the heavy lifting happening in there.      

### Initializing the engine

`AsyncLLM` is a client which wraps the core engine of vLLM. Its core attribute is the `engine_core` which is an instance of `AsyncMPClient`. This object creates a `CoreEngine` which will run an `EngineCore` in a background process. The `EngineCore` is the main component of vLLM, as its name suggests.

<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/v1/engine/core.py</code></figcaption>
{% highlight python %}
47 class EngineCore:
48     """Inner loop of vLLM's Engine."""
100 
100     def __init__(self,
100                  vllm_config: VllmConfig,
100                  executor_class: type[Executor],
100                  log_stats: bool,
100                  executor_fail_callback: Optional[Callable] = None):
100         assert vllm_config.model_config.runner_type != "pooling"
100 
100         logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
100                     VLLM_VERSION, vllm_config)
100 
100         self.log_stats = log_stats
100 
100         # Setup Model.
100         self.model_executor = executor_class(vllm_config)
100         if executor_fail_callback is not None:
100             self.model_executor.register_failure_callback(
100                 executor_fail_callback)
100 
100         # Setup KV Caches and update CacheConfig after profiling.
100         num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
100             self._initialize_kv_caches(vllm_config)
100 
100         vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
100         vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
100 
100         self.structured_output_manager = StructuredOutputManager(vllm_config)
100 
100         # Setup scheduler.
100         if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
100             Scheduler = resolve_obj_by_qualname(
100                 vllm_config.scheduler_config.scheduler_cls)
100         else:
100             Scheduler = vllm_config.scheduler_config.scheduler_cls
100 
100         # This warning can be removed once the V1 Scheduler interface is
100         # finalized and we can maintain support for scheduler classes that
100         # implement it
100         if Scheduler is not V1Scheduler:
100             logger.warning(
100                 "Using configured V1 scheduler class %s. "
100                 "This scheduler interface is not public and "
100                 "compatibility may not be maintained.",
100                 vllm_config.scheduler_config.scheduler_cls)
100 
100         self.scheduler: SchedulerInterface = Scheduler(
100             scheduler_config=vllm_config.scheduler_config,
100             model_config=vllm_config.model_config,
100             cache_config=vllm_config.cache_config,
100             lora_config=vllm_config.lora_config,
100             kv_cache_config=kv_cache_config,
100             speculative_config=vllm_config.speculative_config,
100             structured_output_manager=self.structured_output_manager,
100             include_finished_set=vllm_config.parallel_config.data_parallel_size
100             > 1,
100             log_stats=self.log_stats,
100         )
100 
100         # Setup MM Input Mapper.
100         self.mm_input_cache_server = MirroredProcessingCache(
100             vllm_config.model_config)
100 
100         # Setup batch queue for pipeline parallelism.
100         # Batch queue for scheduled batches. This enables us to asynchronously
100         # schedule and execute batches, and is required by pipeline parallelism
100         # to eliminate pipeline bubbles.
100         self.batch_queue_size = self.model_executor.max_concurrent_batches
100         self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
100                                                      SchedulerOutput]]] = None
100         if self.batch_queue_size > 1:
100             logger.info("Batch queue is enabled with size %d",
100                         self.batch_queue_size)
100             self.batch_queue = queue.Queue(self.batch_queue_size)
{% endhighlight %}   
</figure> 
    
A lot is happening during initialization, among which the KV cache and scheduler setup. We will take about these later, as they are vLLM's key optimizations.   
The `EngineCore` requires an executor to actually run the model. In our case, the executor class is [`MultiProcExecutor`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/multiproc_executor.py), it is in charge of setting up the workers on the device.    
Assuming the user runs the command on a machine with GPUs, it will start several instances of [`GPUWorker`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_worker.py), one for each GPU. Each worker requires a runner for the model, in this case a [`GPUModelRunner`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L62). The runner starts by loading the model on the device thanks a model loader.     
Now, several implementations of the model loader are defined depending on the format of the model weights (`DummyModelLoader`, `TensorizerLoader`, `ShardedStateLoader`, `BitsAndBytesModelLoader`, `GGUFModelLoader`, `RunaiModelStreamerLoader`, `ShardedStateLoader`, `DefaultModelLoader`). The appropriate one will be selected depending on the user's choice and configuration, in the [`get_model_loader`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/loader.py#L1516) function. Let's assume the user does not have a specific configuration and runs the command without any other argument. Hence, the loader will be (`DefaultModelLoader`)[https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/loader.py#L210]. It will get the model path passed in the CLI parameters `'mistralai/Mistral-Small-3.1-24B-Instruct-2503'`. Assuming again that this is the first time the user runs this command on the machine, the loader will download the model weights from Hugging Face Hub ([`download_weights_from_hf`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py#L228)). It will perform a direct API call via the `snapshot_download` function of ` huggingface_hub` library to get the weights.     
The download may take a while depending on the model and the user's bandwidth. In this case, `Mistral-Small-3.1-24B-Instruct-2503` represents about 50Go of safetensors weights. Once the download is complete, the weights will be stored in this folder `~/.cache/huggingface/hub/Mistral-Small-3.1-24B-Instruct-2503` for a faster initialization next time. The worker will then load the weights on the GPU.    
Once the workers are ready, and the core components of the engine are setup, the server will finally start to accept incoming requests.




      
     
#### Webserver
     
The `run_server` function launches vLLM's inference server, built with FastAPI. It initializes a FastAPI application, and binds it to a router. All of that is defined within a context which provides an engine client to the server. This client is stored into the application state at launch and is used during the application's entire lifetime.
       
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>[`vllm/entrypoints/openai/api_server.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py)</code></figcaption>
{% highlight python %}
1041 async def run_server(args, **uvicorn_kwargs) -> None:
...      
1077     async with build_async_engine_client(args) as engine_client:
1078         app = build_app(args)
1079
1080         vllm_config = await engine_client.get_vllm_config()
1081         await init_app_state(engine_client, vllm_config, app.state, args)
{% endhighlight %}   
</figure>    
        
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>[`vllm/entrypoints/openai/api_server.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py)</code></figcaption>
{% highlight python %}
813 def build_app(args: Namespace) -> FastAPI:                                                             
814     if args.disable_fastapi_docs:
815         app = FastAPI(openapi_url=None,
816                       docs_url=None,
817                       redoc_url=None,
818                       lifespan=lifespan)
819     else:
820         app = FastAPI(lifespan=lifespan)
821     app.include_router(router)
822     app.root_path = args.root_path
{% endhighlight %}   
</figure>  
      
The router defines the routes available on the server. These are implemented in the same file.     
vLLM mimics the OpenAI API protocol to be used as a drop-in replacement for applications using OpenAI models. This API protocol requires models to be exposed on the `v1/chat/completions` route. It returns a `StreamingResponse`, indicating that vLLM does support token streaming.    

<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/entrypoints/openai/api_server.py</code></figcaption>
{% highlight python %}
305 router = APIRouter()
...     
466 @router.post("/v1/chat/completions",
467              dependencies=[Depends(validate_json_request)])
468 @with_cancellation
469 @load_aware_call
470 async def create_chat_completion(request: ChatCompletionRequest,
471                                  raw_request: Request):
472     handler = chat(raw_request)
473     if handler is None:
474         return base(raw_request).create_error_response(
475             message="The model does not support Chat Completions API")
476 
477     generator = await handler.create_chat_completion(request, raw_request)
478 
479     if isinstance(generator, ErrorResponse):
480         return JSONResponse(content=generator.model_dump(),
481                             status_code=generator.code)
482 
483     elif isinstance(generator, ChatCompletionResponse):
484         return JSONResponse(content=generator.model_dump())
485 
486     return StreamingResponse(content=generator, media_type="text/event-stream")    
{% endhighlight %}   
</figure>  

Applications based on OpenAI would frequently use the [`ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/) class of LangChain. To switch from OpenAI models to vLLM-served models, the `base_url` parameters should be set to the vLLM's server URL.    
The core logic resides in the `create_chat_completion` function which mimics the OpenAI Chat Completion API. It is a method of the [`OpenAIServingChat`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/serving_chat.py) class, which is instantiated in `init_app_state` and uses the engine client. It relies on the `generate` method of the engine client.   
       
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/entrypoints/openai/serving_chat.py</code></figcaption>
{% highlight python %}
48  class OpenAIServingChat(OpenAIServing):
...
121    async def create_chat_completion(
122        self,
123        request: ChatCompletionRequest,
124        raw_request: Optional[Request] = None,
125    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
126               ErrorResponse]:
...
242         generator = self.engine_client.generate(
243             engine_prompt,
244             sampling_params,
245             request_id,
246             lora_request=lora_request,
247             trace_headers=trace_headers,
248             prompt_adapter_request=prompt_adapter_request,
249             priority=request.priority,
250         )
...
{% endhighlight %}   
</figure>   
           

So we know how vLLM starts its server, how it is built, the route we will call, and when the engine comes to play.         
If you are still here, congratulations !  
Let's leave the server at that. Of course, there could be even more to say, and it is actually a good inspiration if you are trying to build a FastAPI application yourself, but this post is not about FastAPI, it is about vLLM. It is time to talk about the heart of the reactor, the `engine_client`!
     
        
### vLLM engine
     
The inference server is launched within a context that provides an `engine_client` object. It is an asynchronous engine but we do not know yet how it is created. From the code, we see it comes from the `build_async_engine_client` function. Let's take a closer look.
        
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/entrypoints/openai/api_server.py</code></figcaption>
{% highlight python %}
138 @asynccontextmanager
139 async def build_async_engine_client(
140         args: Namespace) -> AsyncIterator[EngineClient]:
141 
142     # Context manager to handle engine_client lifecycle
143     # Ensures everything is shutdown and cleaned up on error/exit
144     engine_args = AsyncEngineArgs.from_cli_args(args)
145 
146     async with build_async_engine_client_from_engine_args(
147             engine_args, args.disable_frontend_multiprocessing) as engine:
148         yield engine
149 
150 
151 @asynccontextmanager
152 async def build_async_engine_client_from_engine_args(
153     engine_args: AsyncEngineArgs,
154     disable_frontend_multiprocessing: bool = False,
155 ) -> AsyncIterator[EngineClient]:
...
168     # V1 AsyncLLM.
169     if envs.VLLM_USE_V1:
...
177         try:
178             async_llm = AsyncLLM.from_vllm_config(
179                 vllm_config=vllm_config,
180                 usage_context=usage_context,
181                 disable_log_requests=engine_args.disable_log_requests,
182                 disable_log_stats=engine_args.disable_log_stats)
183             yield async_llm
... 
188     # V0 AsyncLLM.
189     elif (MQLLMEngineClient.is_unsupported_config(vllm_config)
190           or disable_frontend_multiprocessing):
191 
...
193         try:
194             engine_client = AsyncLLMEngine.from_vllm_config(
195                 vllm_config=vllm_config,
196                 usage_context=usage_context,
197                 disable_log_requests=engine_args.disable_log_requests,
198                 disable_log_stats=engine_args.disable_log_stats)
199             yield engine_client
{% endhighlight %}   
</figure>   
       
OK so this `engine_client` is whether an `AsyncLLM` in V1 or an `AsyncLLMEngine` in V0. Let's assume we are now using V1 as of April 2025. V1 is a significant upgrade of vLLM whose alpha released in January 2025, introducing several optimizations. 
     
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/v1/engine/async_llm.py</code></figcaption>
{% highlight python %}
43  class AsyncLLM(EngineClient):
...        
45        def __init__(
...
55        ) -> None:
...
98            # EngineCore (starts the engine in background process).
99            core_client_class = AsyncMPClient if (
100                vllm_config.parallel_config.data_parallel_size
101                == 1) else DPAsyncMPClient
{% endhighlight %}   
</figure> 
     
The client relies on an `EngineCore` running as a background process. This object implements all the optimizations of vLLM. Soon we should see references to KV cache management in the code. Here, `data_parallel_size` is 1 by default, so the client core is an `AsyncMPClient`. Let's skip a few jumps, this class is an async wrapper around the `MPClient`, (sorry for the spoil), which itself uses `CoreEngine`.
     
<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/v1/engine/core_client.py</code></figcaption>
{% highlight python %}
255 class CoreEngine:
256     """One per data parallel rank."""
257 
258     def __init__(
...
267     ):
...
270         try:
271             # Start EngineCore in background process.
272             self.proc_handle = BackgroundProcHandle(
273                 input_path=input_path,
274                 output_path=output_path,
275                 process_name=f"EngineCore_{index}",
276                 target_fn=EngineCoreProc.run_engine_core,
277                 process_kwargs={
278                     "vllm_config": vllm_config,
279                     "dp_rank": index,
280                     "local_dp_rank": local_dp_rank,
281                     "executor_class": executor_class,
282                     "log_stats": log_stats,
283                 })
{% endhighlight %}   
</figure> 

We do not really care about the `BackgroundProcHandle` here, it is a technical object to run a procedure in the background. We are more interested in the procedure it is running, which is `EngineCoreProc.run_engine_core`. This procedure is an instance of `EngineCoreProc` which is an `EngineCore`

<figure class="custom-code-block">
  <figcaption style="margin-bottom: 0.2em;"><code>vllm/v1/engine/core.py</code></figcaption>
{% highlight python %}
47 class EngineCore:
48     """Inner loop of vLLM's Engine."""
100 
100     def __init__(self,
100                  vllm_config: VllmConfig,
100                  executor_class: type[Executor],
100                  log_stats: bool,
100                  executor_fail_callback: Optional[Callable] = None):
100         assert vllm_config.model_config.runner_type != "pooling"
100 
100         logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
100                     VLLM_VERSION, vllm_config)
100 
100         self.log_stats = log_stats
100 
100         # Setup Model.
100         self.model_executor = executor_class(vllm_config)
100         if executor_fail_callback is not None:
100             self.model_executor.register_failure_callback(
100                 executor_fail_callback)
100 
100         # Setup KV Caches and update CacheConfig after profiling.
100         num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
100             self._initialize_kv_caches(vllm_config)
100 
100         vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
100         vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
100 
100         self.structured_output_manager = StructuredOutputManager(vllm_config)
100 
100         # Setup scheduler.
100         if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
100             Scheduler = resolve_obj_by_qualname(
100                 vllm_config.scheduler_config.scheduler_cls)
100         else:
100             Scheduler = vllm_config.scheduler_config.scheduler_cls
100 
100         # This warning can be removed once the V1 Scheduler interface is
100         # finalized and we can maintain support for scheduler classes that
100         # implement it
100         if Scheduler is not V1Scheduler:
100             logger.warning(
100                 "Using configured V1 scheduler class %s. "
100                 "This scheduler interface is not public and "
100                 "compatibility may not be maintained.",
100                 vllm_config.scheduler_config.scheduler_cls)
100 
100         self.scheduler: SchedulerInterface = Scheduler(
100             scheduler_config=vllm_config.scheduler_config,
100             model_config=vllm_config.model_config,
100             cache_config=vllm_config.cache_config,
100             lora_config=vllm_config.lora_config,
100             kv_cache_config=kv_cache_config,
100             speculative_config=vllm_config.speculative_config,
100             structured_output_manager=self.structured_output_manager,
100             include_finished_set=vllm_config.parallel_config.data_parallel_size
100             > 1,
100             log_stats=self.log_stats,
100         )
100 
100         # Setup MM Input Mapper.
100         self.mm_input_cache_server = MirroredProcessingCache(
100             vllm_config.model_config)
100 
100         # Setup batch queue for pipeline parallelism.
100         # Batch queue for scheduled batches. This enables us to asynchronously
100         # schedule and execute batches, and is required by pipeline parallelism
100         # to eliminate pipeline bubbles.
100         self.batch_queue_size = self.model_executor.max_concurrent_batches
100         self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
100                                                      SchedulerOutput]]] = None
100         if self.batch_queue_size > 1:
100             logger.info("Batch queue is enabled with size %d",
100                         self.batch_queue_size)
100             self.batch_queue = queue.Queue(self.batch_queue_size)
{% endhighlight %}   
</figure> 
      
Jackpot! Everything is here: the kv cache, the scheduler, the GPU and CPU blocks, etc. If you look at the methods there are even decoding steps and incoming request management! This is indeed the core of vLLM V1 engine. There is just one more minor question...       
           
                    
                  
What does this all mean?

## Cracking open vLLM engine's optimizations

### The problem with naïve KV cache management

In my [previous post]({{ site.url }}/peeks/llm-inference-engines-servers/), I explained how LLM inference is computation-bound. In details, it consists of two phases:
- Prefill-phase (*Prompt processing*): Compute Q, K, V matrices for each token in the prompt. This phase is **compute-bound**. However, computations here can be run in parallel, because all tokens are known, so it is usually fast.     
      
*e.g. it is the time before the first token appears in ChatGPT*.
- Decoding phase (*Generation*): Generate tokens one by one from the prompt, hence this phase is sequential. Computations cannot be run in parallel and K, V matrices for each new token shall be cached to avoid recomputing them at every step. The bottleneck here is managing the linear growth of KV cache with sequence length, hence it is a **memory-bound** process. This phase is usually much slower.      
       
*e.g. it is the time from when the first token appears until the answer is complete*

Because the decoding phase usually dominates latency, we often summarize LLM inference as **memory-bound**.   
    
Managing the KV cache is the key issue here.    
First, as a rule of thumb, model weights take ~60% of GPU memory, and the KV cache ~30%, leaving ~10% for computations during decoding. A naïve implementation would always allocate the maximum memory available, regardless of the sequence length. Since most tensor processing frameworks require tensors to be stored in contiguous memory, older serving frameworks would allocate a contiguous chunk of GPU memory for caching. It would typically be the largest possible size, which is the context window of the model. Context windows went from 2048 tokens at first in open-source LLM, to up to [10 million with Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/#:~:text=Llama%204%20Scout%20offers%20an%20industry%2Dleading%20context%20window%20of%2010M) now. So, this method scales poorly.    
Then, sequence lengths are not known in advance. Hence, allocating a huge chunk of memory for a request that will eventually use just a small fraction of it is highly inefficient, as a large part will remain unused, *and* unavailable to other incoming sequences. Pre-allocation is not suited for the dynamic nature of decoding.    
Finally, LLMs do not generate one sequence only. It might be surprising, because you only see one answer in the end, but LLMs use decoding algorithms like beam search, which run several decoding in parallel, resulting in multiple output sequences. These sequences may often include the same first tokens until they diverge. In older frameworks, the matrices for these common tokens would be cached several times in different places, because they would have to be stored in contiguous memory space.     
        
To tackle this issue, vLLM introduces **PagedAttention**, in addition with a **near-zero waste KV cache manager**, and an **efficient sequence scheduler**.

### vLLM's KV cache manager    

Instead of pre-allocating one large contiguous chunk of memory, vLLM's KV cache manager splits it in small contiguous memory blocks. These are fixed-size but not necessarily contiguous. A block shall be small enough that it would waste only a tiny fraction of memory if unused, yet large enough to hold at least one semantically meaningful item. The smallest item is a (K, V) pair for one token, so a block may hold at least one pair, hence we could say blocks are *token-granular*. The engine allocates blocks dynamically as the sequence grows.
However, this approach alone would not enable sharing of token matrices between multiple sequences during decoding, nor would it suit traditional tensor-processing frameworks expecting one big contiguous tensor.    
The core idea behind vLLM's KV cache manager is to use *logical* blocks that map to *physical* ones. This is inspired by virtual memory in operating systems. Distinct logical blocks can point to the same physical blocks to enable sharing. Tensor-processing frameworks would also perceive logical blocks as one large contiguous tensor.
To make it work, the engine must maintain a *mapping table* to keep track of block references for each sequence. Much like how a Spark driver would keep track of which executor holds which RDD partition of a dataset, across a distributed system.
This logical-to-physical mapping is the key idea of vLLM. It enables the KV cache to grow dynamically, without pre-allocating physical memory, and to share matrices across multiple sequences during decoding. Virtual blocks may be pre-allocated, but not take any actual space on GPU, until physical blocks are allocated and mapped by the engine.
       
      
### PagedAttention

To understand PagedAttention, let's first come back to the self-attention equation:       
        
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$     

Focus on what happens at decoding step $t$ (for the first layer and one attention head):     
1. Get the embedding of the current token $x_t \in \mathbb{R}^{d_e}$, where $d_e$ is the embedding size.
2. Project the embedding to get the current query, key, and value vectors, with $d$ the projection dimension       
$$
\begin{gather}
  q_t = W_q x_t \in \mathbb{R}^{d},\\
  k_t = W_k x_t \in \mathbb{R}^{d},\\
  v_t = W_v x_t \in \mathbb{R}^{d}.
\end{gather}
$$      
3. Concatenate the current key and value vectors $k_t, v_t$ with previous ones $(k_1,...,k_{t-1})$ and $(v_1,...,v_{t-1})$ to get $K_t, V_t \in \mathbb{R}^{t \times d}$. Retrieve the previous vectors from KV-cache if available, otherwise recompute them.
4. Compute the attention scores for the current token $a_t = \text{softmax}\left(\frac{q_t K_t^\top}{\sqrt{d}}\right) \in \mathbb{R}^t$.
5. Compute the layer output $o_t = a_t V_t \in \mathbb{R}^d$.
      
For the next layers, the steps are essentially the same except the input at step 1 is the output of the previous layer.   
         
However, this does not work with non contiguous blocks, because tensor-processing frameworks such as PyTorch cannot perform the dot products of steps 4 and 5.     
In vLLM, key and value vectors of previous steps for one sequence are stored in blocks in the KV-cache. For each block $j$ holding $B$ tokens, we have $K_j$ and $V_j$, the concatenations of contiguous key vectors, and contiguous value vectors, both in $\mathbb{R}^{B \times d}$.    
A block being a contiguous memory chunk, we may compute the dot-product $s_j=\text{exp}\left(\frac{q_t K_j^\top}{\sqrt{d}}\right) \in \mathbb{R}^{1 \times B}$ and then sum these for each block $i$ in the sequence to get the block-wise attention score $A_j = \frac{s_j}{\sum_i s_i} \in \mathbb{R}^{1 \times B}$. Finally, we sum the partial weighted sums of value vectors of each block, to get the final output, $o_t = \sum_j A_j V_j \in \mathbb{R}^{1 \times d}$.     
This is **PagedAttention**, a block-wise decomposition of the self attention equation.
               
vLLM developped specific kernels, *i.e functions to perform computations on GPU*, in order to implement PagedAttention efficiently.
      

### Sequence scheduler

In real-life conditions, we would process sequences in batches to maximize GPU utilization and minimize latency for users.      
A naïve implementation would pre-allocate the maximum possible output length for each sequence in the batch. This would lead to huge memory wastes, as the majority of sequences are likely to complete way before reaching this limit. With current context windows of up to 10M tokens, it would simply be impossible to allocate enough memory. vLLM avoids this pitfall with logical-to-physical block mapping and PagedAttention, leaving more memory for sequences to be processed.    
Naïvely scheduling batches at sequence-level would still be very inefficient. This would mean waiting for enough sequences to arrive to process a batch, and new incoming sequences would have to wait for the whole batch to finish. Hence, vLLM's scheduler implements iteration-level scheduling. At each decoding step, the scheduler removes completed sequences from the batch and add new ones, always trying to maximize GPU utilization. This results in what is called   **continuous batching**, the batch size varies with incoming and outcoming sequences (you may also see **in-flight** or **dynamic batching**, these are the same thing).    
These optimizations achieve near-optimal memory usage and significantly increase vLLM's throughtput in batch processing.       
      
Now what happens when there are simply too many incoming sequences?     
In this case, vLLM implements a first-in-first-out type policy. Low priority sequences may be evicted from GPU if it is at capacity. In this case, their KV-cache blocks are sent to CPU to await processing. Once earlier sequences complete, the evicted sequence's blocks are sent back to GPU to resume generation. vLLM does not accept new incoming sequences until evicted sequences are swapped back and completed.  
Thus CPU memory space is also bounded, and some evicted sequences' blocks are simply deleted when running out of CPU RAM. In this case, the KV cache for these sequences is recomputed when GPU memory is available again.    
An evicted sequence is composed of the initial prompt plus tokens generated before eviction. As all tokens are known, it can be considered as one larger prompt and KV cache may be recomputed in parallel, similar to the pre-fill phase. Although it still adds latency, it is much faster than sequential processing during decoding.

#TODO: vLLM has been integrated in Hugging Face Hub