"""Minimal client for batch querying vLLM server."""

from typing import List, Optional, Dict, Any
from openai import OpenAI, AsyncOpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re


class VLLMClient:
    """Client for interacting with vLLM server via OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: float = 60.0
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: Base URL of the vLLM server (default: http://localhost:8000/v1)
            api_key: API key (not needed for local vLLM, but required by OpenAI client)
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        self._cached_model = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Warning: Could not fetch models from server: {e}")
            return []
    
    def get_default_model(self) -> Optional[str]:
        """Get the default model from the server (first available model)."""
        if self._cached_model is None:
            models = self.get_available_models()
            if models:
                self._cached_model = models[0]
            else:
                # Fallback to a common default
                self._cached_model = "microsoft/phi-2"
        return self._cached_model
    
    def generate(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        return_full_response: bool = False,
        prefill: Optional[str] = None,
        **kwargs
    ) -> List[Any]:
        """
        Generate completions for a batch of prompts using parallel processing.
        
        Args:
            prompts: List of input prompts
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            return_full_response: If True, returns full response dict with all metadata
            prefill: Optional text to prefill the assistant's response (model continues from here)
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions (str) or full response dicts if return_full_response=True
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single(prompt: str) -> Any:
            """Generate completion for a single prompt."""
            # Build messages array
            messages = [{"role": "user", "content": prompt}]
            
            # Add assistant prefill if provided
            if prefill:
                messages.append({"role": "assistant", "content": prefill})
            
            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            response = self.client.chat.completions.create(**request_params)
            
            if return_full_response:
                # Return complete response as dict
                completion_text = response.choices[0].message.content or ""
                prefill_text = prefill if prefill else ""
                
                result = {
                    "prompt": prompt,
                    "prefill": prefill if prefill else None,
                    "completion": completion_text,
                    "full_text": prefill_text + completion_text,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                }
                
                # Include additional fields if present (e.g., reasoning tokens, logprobs)
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    result["logprobs"] = response.choices[0].logprobs
                
                # For models that support reasoning/thinking (like o1)
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    result["reasoning"] = response.choices[0].message.reasoning_content
                
                # Include the raw response object for full access
                result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
                
                return result
            else:
                # Return full text (prefill + completion) if prefill was used
                completion_text = response.choices[0].message.content or ""
                if prefill:
                    return prefill + completion_text
                return completion_text
        
        # Process prompts in parallel using ThreadPoolExecutor
        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = f"Error: {str(e)}"
        
        return responses
    
    def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a batch of prompts with full response details using parallel processing.
        
        Args:
            prompts: List of input prompts
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of response dictionaries with full details
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single_detailed(prompt: str) -> Dict[str, Any]:
            """Generate completion with full details for a single prompt."""
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            result = {
                "prompt": prompt,
                "completion": response.choices[0].message.content or "",
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            }
            
            # Include additional fields if present
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                result["logprobs"] = response.choices[0].logprobs
            
            # For models that support reasoning/thinking
            if hasattr(response.choices[0].message, 'reasoning_content'):
                result["reasoning"] = response.choices[0].message.reasoning_content
            
            # Include the raw response object for complete access
            result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
            
            return result
        
        # Process prompts in parallel using ThreadPoolExecutor
        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single_detailed, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = {
                        "prompt": prompts[idx],
                        "completion": f"Error: {str(e)}",
                        "model": model,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
        
        return responses
    
    def generate_with_messages(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        return_full_response: bool = False,
        **kwargs
    ) -> List[Any]:
        """
        Generate completions using custom message arrays for full control.
        This allows multi-turn conversations, system prompts, and prefilling.
        
        Args:
            messages_list: List of message arrays, where each array contains dicts with 'role' and 'content'
                          Example: [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]]
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            return_full_response: If True, returns full response dict with all metadata
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions (str) or full response dicts if return_full_response=True
            
        Example:
            # Prefill assistant response
            messages = [
                {"role": "user", "content": "Write a story about a cat"},
                {"role": "assistant", "content": "Once upon a time, there was a cat named"}
            ]
            results = client.generate_with_messages([messages], max_tokens=100)
        """
        if not messages_list:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single(messages: List[Dict[str, str]]) -> Any:
            """Generate completion for a single message array."""
            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            response = self.client.chat.completions.create(**request_params)
            
            if return_full_response:
                # Extract prefill if the last message was from assistant
                prefill = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else None
                completion_text = response.choices[0].message.content or ""
                prefill_text = prefill if prefill else ""
                
                result = {
                    "messages": messages,
                    "prefill": prefill,
                    "completion": completion_text,
                    "full_text": prefill_text + completion_text,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                }
                
                # Include additional fields if present
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    result["logprobs"] = response.choices[0].logprobs
                
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    result["reasoning"] = response.choices[0].message.reasoning_content
                
                result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
                
                return result
            else:
                return response.choices[0].message.content or ""
        
        # Process messages in parallel using ThreadPoolExecutor
        responses = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single, messages): idx 
                for idx, messages in enumerate(messages_list)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = f"Error: {str(e)}"
        
        return responses
    
    async def generate_async(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for a batch of prompts using async processing (faster for many requests).
        
        Args:
            prompts: List of input prompts
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        async_client = AsyncOpenAI(
            base_url=self.client.base_url,
            api_key=self.client.api_key,
            timeout=self.client.timeout
        )
        
        async def _generate_single_async(prompt: str) -> str:
            """Generate completion for a single prompt asynchronously."""
            response = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        
        # Process all prompts concurrently
        tasks = [_generate_single_async(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for resp in responses:
            if isinstance(resp, Exception):
                results.append(f"Error: {str(resp)}")
            else:
                results.append(resp)
        
        await async_client.close()
        return results
    
    def generate_raw(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        return_full_response: bool = False,
        prefill: Optional[str] = None,
        **kwargs
    ) -> List[Any]:
        """
        Generate completions using raw prompts (bypasses chat templates).
        This gives you full control over the prompt format, including reasoning structure.
        
        Args:
            prompts: List of input prompts (full prompt text, not just user messages)
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            return_full_response: If True, returns full response dict with all metadata
            prefill: Optional text to prefill (will be appended to prompt before generation)
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions (str) or full response dicts if return_full_response=True
            
        Example:
            # Create a raw prompt with reasoning structure for gpt-oss-20b
            prompt = "User: What is 2+2?\nAssistant: <reasoning>\nLet me think about this..."
            prefill = "The answer is obviously"
            result = client.generate_raw([prompt], prefill=prefill, max_tokens=100)
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single(prompt: str) -> Any:
            """Generate completion for a single raw prompt."""
            # Append prefill to prompt if provided
            full_prompt = prompt
            if prefill:
                full_prompt = prompt + prefill
            
            # Use completions endpoint (not chat.completions) to bypass chat templates
            request_params = {
                "model": model,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            # Use the completions endpoint for raw prompt control
            response = self.client.completions.create(**request_params)
            
            if return_full_response:
                completion_text = response.choices[0].text or ""
                prefill_text = prefill if prefill else ""
                
                # Try to extract reasoning from completion text for models like gpt-oss-20b
                # The completion continues from where the prompt left off (after <|channel|>analysis<|message|>reasoning_prefill)
                # So completion_text may contain: more_reasoning + possibly <|channel|>final<|message|>final_answer
                
                # First, extract any reasoning prefill from the prompt
                reasoning_prefill_from_prompt = ""
                if "<|channel|>analysis<|message|>" in prompt:
                    # Extract text after the analysis marker in the prompt (this is the prefill)
                    prompt_parts = prompt.split("<|channel|>analysis<|message|>")
                    if len(prompt_parts) > 1:
                        reasoning_prefill_from_prompt = prompt_parts[1]
                
                # Parse the completion to separate reasoning continuation from final answer
                # Note: Special tokens like <|channel|>final<|message|> may be decoded/stripped by vLLM
                # and appear as "assistantfinal" in the text, so we need to handle both cases
                reasoning_continuation = None
                final_answer = None
                
                # Try multiple patterns for finding the final answer marker
                # 1. Proper channel marker
                if "<|channel|>final<|message|>" in completion_text:
                    parts = completion_text.split("<|channel|>final<|message|>")
                    reasoning_continuation = parts[0]
                    final_answer = parts[1] if len(parts) > 1 else ""
                # 2. Decoded/stripped special token appearing as "assistantfinal"
                elif "assistantfinal" in completion_text.lower():
                    # Find the first occurrence (should only be one, but be safe)
                    # Use case-insensitive search to find where to split
                    match = re.search(r'assistantfinal', completion_text, flags=re.IGNORECASE)
                    if match:
                        split_idx = match.start()
                        reasoning_continuation = completion_text[:split_idx]
                        final_answer = completion_text[match.end():] if match.end() < len(completion_text) else ""
                    else:
                        # Fallback: use split
                        parts = re.split(r'assistantfinal', completion_text, flags=re.IGNORECASE, maxsplit=1)
                        reasoning_continuation = parts[0]
                        final_answer = parts[1] if len(parts) > 1 else ""
                # 3. Other channel markers
                elif "<|channel|>" in completion_text:
                    next_channel_idx = completion_text.find("<|channel|>")
                    reasoning_continuation = completion_text[:next_channel_idx]
                    final_answer = completion_text[next_channel_idx:]
                # 4. No markers found - all reasoning
                else:
                    reasoning_continuation = completion_text
                    final_answer = None
                
                # Combine prefill with continuation to get full reasoning
                if reasoning_prefill_from_prompt:
                    reasoning_text = reasoning_prefill_from_prompt + reasoning_continuation
                else:
                    reasoning_text = reasoning_continuation
                
                # Final cleanup: ensure reasoning doesn't contain "assistantfinal" (shouldn't happen, but be safe)
                if reasoning_text and "assistantfinal" in reasoning_text.lower():
                    # Remove everything from "assistantfinal" onwards from reasoning
                    match = re.search(r'assistantfinal', reasoning_text, flags=re.IGNORECASE)
                    if match:
                        reasoning_text = reasoning_text[:match.start()]
                
                result = {
                    "prompt": prompt,
                    "prefill": prefill if prefill else None,
                    "completion": final_answer if final_answer is not None else completion_text,
                    "reasoning": reasoning_text,
                    "full_text": prefill_text + completion_text,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                }
                
                # Include additional fields if present
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    result["logprobs"] = response.choices[0].logprobs
                
                result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
                
                return result
            else:
                # Return full text (prefill + completion) if prefill was used
                completion_text = response.choices[0].text or ""
                if prefill:
                    return prefill + completion_text
                return completion_text
        
        # Process prompts in parallel using ThreadPoolExecutor
        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = f"Error: {str(e)}"
        
        return responses


def format_prompt(task: str, rules: str, grader: str) -> str:
    """Format a reward hacking prompt for the model."""
    prompt = f"{task}\n\n{rules}\n\n{grader}"
    return prompt


def format_prompts_batch(
    tasks: List[str],
    rules: List[str],
    graders: List[str]
) -> List[str]:
    """
    Efficiently format a batch of reward hacking prompts.
    
    Args:
        tasks: List of task descriptions
        rules: List of rule descriptions
        graders: List of grader code blocks
        
    Returns:
        List of formatted prompts
    """
    if not (len(tasks) == len(rules) == len(graders)):
        raise ValueError("All input lists must have the same length")
    
    # Use list comprehension for efficient batch formatting
    prompts = [
        f"{task}\n\n{rule}\n\n{grader}"
        for task, rule, grader in zip(tasks, rules, graders)
    ]
    
    return prompts


def format_prompts_from_dicts(prompt_dicts: List[Dict[str, str]]) -> List[str]:
    """
    Format prompts from a list of dictionaries containing 'task', 'rules', and 'grader' keys.
    
    Args:
        prompt_dicts: List of dictionaries with 'task', 'rules', and 'grader' keys
        
    Returns:
        List of formatted prompts
    """
    tasks = [d["task"] for d in prompt_dicts]
    rules = [d["rules"] for d in prompt_dicts]
    graders = [d["grader"] for d in prompt_dicts]
    
    return format_prompts_batch(tasks, rules, graders)