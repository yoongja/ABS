import os
import openai
import backoff
from dotenv import load_dotenv
import logging
load_dotenv()
completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, (openai.error.OpenAIError, TimeoutError, openai.error.RateLimitError), max_time=10)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=128, n=1, logprobs=True, top_logprobs=5, stop=None) -> list:

    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, logprobs=logprobs, top_logprobs=top_logprobs,stop=stop)

def chatgpt(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=128, n=1, logprobs=True, top_logprobs=5, stop=None) -> list:
    """
    ChatGPT wrapper for OpenAI API calls.
    """
    # breakpoint()
    global completion_tokens, prompt_tokens
    outputs = []

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": cnt,
            "stop": stop,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs
        }
        try:
            # Perform the API call with backoff
            res = completions_with_backoff(**request_params)
            
            # Log successful API call
            logging.info("API call successful.")
            logging.info(f"Response: {res}")
            
            if "choices" in res:
                outputs = ([choice for choice in res['choices']])
                
            # Log token usage
            completion_tokens += res["usage"]["completion_tokens"]
            prompt_tokens += res["usage"]["prompt_tokens"]
            logging.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")

        except Exception as e:
            # Log any exceptions during API call
            logging.error(f"API call failed: {e}")
            raise  # Re-raise the exception after logging

    # Log the final outputs
    logging.info(f"Final outputs: {outputs}")
    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.025 + prompt_tokens / 1000 * 0.1
    elif backend == "gpt-4o-mini":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
