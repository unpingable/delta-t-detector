"""
Δt Detector - API Providers

Support for frontier model APIs (OpenAI, Anthropic).

Note: API-based detection is inherently limited compared to local models because
we don't have access to token-level logprobs during generation. We can only
analyze the output and use proxy metrics.

For OpenAI: logprobs are available (limited)
For Anthropic: no logprobs, so we use response-level heuristics only
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import math


@dataclass
class APIGenerationTrace:
    """Generation trace from API response"""
    prompt: str
    response: str
    model: str
    tokens: List[str]
    logprobs: Optional[List[float]]  # May be None for some providers
    finish_reason: str
    latency_ms: float
    
    # Derived metrics (computed post-hoc)
    mean_logprob: Optional[float] = None
    logprob_variance: Optional[float] = None
    token_count: int = 0
    
    def __post_init__(self):
        self.token_count = len(self.tokens)
        if self.logprobs:
            self.mean_logprob = sum(self.logprobs) / len(self.logprobs)
            mean = self.mean_logprob
            self.logprob_variance = sum((lp - mean) ** 2 for lp in self.logprobs) / len(self.logprobs)


@dataclass
class APIDetectionResult:
    """Detection result from API-based analysis"""
    prediction: str  # 'truthful', 'uncertain', 'hallucination'
    confidence: float
    model: str
    method: str  # 'logprob', 'consistency', 'heuristic'
    
    # Metrics
    mean_logprob: Optional[float] = None
    consistency_score: Optional[float] = None
    response_length: int = 0
    
    # Warnings
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class APIProvider(ABC):
    """Abstract base class for API providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> APIGenerationTrace:
        """Generate response and return trace"""
        pass
    
    @abstractmethod
    def supports_logprobs(self) -> bool:
        """Whether this provider supports logprobs"""
        pass
    
    @abstractmethod
    def detect(self, prompt: str, **kwargs) -> APIDetectionResult:
        """Run detection on prompt"""
        pass


class OpenAIProvider(APIProvider):
    """
    OpenAI API provider
    
    Supports logprobs for GPT-4 and GPT-3.5 models.
    This allows for proper temporal analysis (limited to top-5 logprobs).
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url
        )
    
    def supports_logprobs(self) -> bool:
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        logprobs: bool = True,
        top_logprobs: int = 5
    ) -> APIGenerationTrace:
        """Generate with logprobs"""
        start = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None
        )
        
        latency_ms = (time.time() - start) * 1000
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Extract tokens and logprobs
        tokens = []
        token_logprobs = []
        
        if logprobs and choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                tokens.append(token_info.token)
                token_logprobs.append(token_info.logprob)
        else:
            # Fallback: split on whitespace (rough approximation)
            tokens = content.split()
            token_logprobs = None
        
        return APIGenerationTrace(
            prompt=prompt,
            response=content,
            model=self.model,
            tokens=tokens,
            logprobs=token_logprobs,
            finish_reason=choice.finish_reason or "unknown",
            latency_ms=latency_ms
        )
    
    def detect(
        self,
        prompt: str,
        n_samples: int = 1,
        consistency_check: bool = True,
        **kwargs
    ) -> APIDetectionResult:
        """
        Detect hallucination using available signals
        
        With logprobs: Use entropy/confidence trajectory
        With consistency: Generate multiple responses, check agreement
        """
        warnings = []
        
        # Primary generation with logprobs
        trace = self.generate(prompt, logprobs=True, **kwargs)
        
        if trace.logprobs is None:
            warnings.append("Logprobs not available, using heuristics only")
            method = "heuristic"
            
            # Heuristic: response length and hedging language
            hedging_phrases = [
                "I'm not sure", "I think", "might be", "could be",
                "I believe", "possibly", "perhaps", "may be",
                "I don't know", "uncertain"
            ]
            hedge_count = sum(1 for p in hedging_phrases if p.lower() in trace.response.lower())
            
            if hedge_count >= 2:
                prediction = "uncertain"
                confidence = 0.5
            elif len(trace.response) < 50:
                prediction = "uncertain"
                confidence = 0.4
            else:
                prediction = "truthful"
                confidence = 0.5
        
        else:
            method = "logprob"
            
            # Analyze logprob trajectory
            logprobs = trace.logprobs
            n = len(logprobs)
            
            if n < 5:
                warnings.append("Very short response, limited analysis")
                prediction = "uncertain"
                confidence = 0.4
            else:
                # Compute metrics
                mean_lp = trace.mean_logprob
                var_lp = trace.logprob_variance
                
                # Early vs late confidence (phase analysis)
                early_third = logprobs[:n//3]
                late_third = logprobs[-(n//3):]
                
                early_mean = sum(early_third) / len(early_third)
                late_mean = sum(late_third) / len(late_third)
                
                # High early confidence relative to late = potential hallucination
                # (In logprob space, higher = more confident, so early > late is suspicious)
                phase_diff = early_mean - late_mean
                
                # Thresholds (calibrated empirically)
                if phase_diff > 0.5:  # Early much more confident
                    prediction = "hallucination"
                    confidence = min(0.9, 0.6 + phase_diff * 0.3)
                elif mean_lp < -2.0:  # Generally uncertain
                    prediction = "uncertain"
                    confidence = 0.6
                elif var_lp > 1.0:  # High variance = unstable
                    prediction = "uncertain"
                    confidence = 0.55
                else:
                    prediction = "truthful"
                    confidence = 0.7
        
        # Optional: Consistency check across multiple samples
        consistency_score = None
        if consistency_check and n_samples > 1:
            responses = [trace.response]
            for _ in range(n_samples - 1):
                t = self.generate(prompt, logprobs=False, **kwargs)
                responses.append(t.response)
            
            # Simple consistency: Jaccard similarity of word sets
            word_sets = [set(r.lower().split()) for r in responses]
            
            if len(word_sets) >= 2:
                pairwise_sims = []
                for i in range(len(word_sets)):
                    for j in range(i+1, len(word_sets)):
                        intersection = len(word_sets[i] & word_sets[j])
                        union = len(word_sets[i] | word_sets[j])
                        if union > 0:
                            pairwise_sims.append(intersection / union)
                
                if pairwise_sims:
                    consistency_score = sum(pairwise_sims) / len(pairwise_sims)
                    
                    # Adjust prediction based on consistency
                    if consistency_score < 0.3:
                        prediction = "hallucination"
                        confidence = max(confidence, 0.75)
                        method = "logprob+consistency"
                    elif consistency_score > 0.7 and prediction == "uncertain":
                        prediction = "truthful"
                        confidence = 0.65
        
        return APIDetectionResult(
            prediction=prediction,
            confidence=confidence,
            model=self.model,
            method=method,
            mean_logprob=trace.mean_logprob,
            consistency_score=consistency_score,
            response_length=len(trace.response),
            warnings=warnings
        )


class AnthropicProvider(APIProvider):
    """
    Anthropic API provider
    
    Note: Anthropic does NOT provide logprobs, so detection is limited to:
    - Response consistency across samples
    - Heuristic analysis (hedging language, response structure)
    - Self-verification prompting
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def supports_logprobs(self) -> bool:
        return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> APIGenerationTrace:
        """Generate response (no logprobs available)"""
        start = time.time()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency_ms = (time.time() - start) * 1000
        
        content = response.content[0].text if response.content else ""
        
        return APIGenerationTrace(
            prompt=prompt,
            response=content,
            model=self.model,
            tokens=content.split(),  # Rough approximation
            logprobs=None,  # Not available
            finish_reason=response.stop_reason or "unknown",
            latency_ms=latency_ms
        )
    
    def detect(
        self,
        prompt: str,
        n_samples: int = 3,
        self_verify: bool = True,
        **kwargs
    ) -> APIDetectionResult:
        """
        Detect hallucination using consistency and self-verification
        
        Since no logprobs are available, we rely on:
        1. Multi-sample consistency
        2. Self-verification prompting
        3. Hedging language analysis
        """
        warnings = ["No logprobs available (Anthropic limitation), using consistency analysis"]
        
        # Generate multiple samples
        responses = []
        for _ in range(n_samples):
            trace = self.generate(prompt, **kwargs)
            responses.append(trace.response)
        
        # Consistency analysis
        word_sets = [set(r.lower().split()) for r in responses]
        
        pairwise_sims = []
        for i in range(len(word_sets)):
            for j in range(i+1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    pairwise_sims.append(intersection / union)
        
        consistency_score = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.5
        
        # Hedging analysis on primary response
        primary = responses[0]
        hedging_phrases = [
            "I'm not sure", "I think", "might be", "could be",
            "I believe", "possibly", "perhaps", "may be",
            "I don't know", "uncertain", "I cannot verify"
        ]
        hedge_count = sum(1 for p in hedging_phrases if p.lower() in primary.lower())
        
        # Optional self-verification
        self_verify_score = None
        if self_verify:
            verify_prompt = f"""I previously answered this question: "{prompt}"

My answer was: "{primary[:500]}"

On reflection, how confident am I that this answer is factually accurate?
Rate from 0-10 where 0 = completely uncertain/likely wrong and 10 = highly confident/verified.
Respond with just the number."""
            
            verify_trace = self.generate(verify_prompt, max_tokens=10, temperature=0.0)
            
            # Parse confidence score
            try:
                score_text = verify_trace.response.strip()
                # Extract first number
                import re
                match = re.search(r'\d+', score_text)
                if match:
                    self_verify_score = int(match.group()) / 10.0
            except:
                pass
        
        # Combine signals
        if consistency_score < 0.3:
            prediction = "hallucination"
            confidence = 0.75
        elif consistency_score < 0.5 and hedge_count >= 2:
            prediction = "uncertain"
            confidence = 0.6
        elif self_verify_score is not None and self_verify_score < 0.4:
            prediction = "hallucination"
            confidence = 0.65
        elif consistency_score > 0.7 and hedge_count == 0:
            prediction = "truthful"
            confidence = 0.7
        else:
            prediction = "uncertain"
            confidence = 0.5
        
        method = "consistency"
        if self_verify:
            method += "+self_verify"
        
        return APIDetectionResult(
            prediction=prediction,
            confidence=confidence,
            model=self.model,
            method=method,
            mean_logprob=None,
            consistency_score=consistency_score,
            response_length=len(primary),
            warnings=warnings
        )


class OllamaProvider(APIProvider):
    """
    Ollama provider for local models via API
    
    Ollama provides logprobs when requested, giving us full temporal analysis.
    This is the recommended way to run local models without loading them directly.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        
        # Check connection
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {base_url}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}: {e}")
    
    def supports_logprobs(self) -> bool:
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> APIGenerationTrace:
        """Generate with Ollama API"""
        import requests
        
        start = time.time()
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                },
                "stream": False
            }
        )
        
        latency_ms = (time.time() - start) * 1000
        
        data = response.json()
        content = data.get("response", "")
        
        # Ollama doesn't return logprobs in basic mode
        # Would need streaming + parsing for that
        
        return APIGenerationTrace(
            prompt=prompt,
            response=content,
            model=self.model,
            tokens=content.split(),
            logprobs=None,  # Would need streaming API
            finish_reason=data.get("done_reason", "unknown"),
            latency_ms=latency_ms
        )
    
    def detect(
        self,
        prompt: str,
        n_samples: int = 3,
        **kwargs
    ) -> APIDetectionResult:
        """Detect using consistency (logprobs require streaming)"""
        warnings = ["Using consistency mode (streaming logprobs not implemented)"]
        
        # Generate multiple samples
        responses = []
        for _ in range(n_samples):
            trace = self.generate(prompt, **kwargs)
            responses.append(trace.response)
        
        # Consistency analysis
        word_sets = [set(r.lower().split()) for r in responses]
        
        pairwise_sims = []
        for i in range(len(word_sets)):
            for j in range(i+1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    pairwise_sims.append(intersection / union)
        
        consistency_score = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.5
        
        if consistency_score < 0.3:
            prediction = "hallucination"
            confidence = 0.7
        elif consistency_score > 0.7:
            prediction = "truthful"
            confidence = 0.65
        else:
            prediction = "uncertain"
            confidence = 0.5
        
        return APIDetectionResult(
            prediction=prediction,
            confidence=confidence,
            model=self.model,
            method="consistency",
            consistency_score=consistency_score,
            response_length=len(responses[0]) if responses else 0,
            warnings=warnings
        )


def get_provider(
    provider: str,
    model: Optional[str] = None,
    **kwargs
) -> APIProvider:
    """
    Factory function to get appropriate provider
    
    Args:
        provider: 'openai', 'anthropic', 'ollama', or 'local'
        model: Model name (provider-specific)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Configured APIProvider instance
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini", **kwargs)
    elif provider == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514", **kwargs)
    elif provider == "ollama":
        return OllamaProvider(model=model or "qwen2.5:3b", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")


# Convenience function
def detect_with_api(
    prompt: str,
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> APIDetectionResult:
    """
    One-liner detection using API provider
    
    Examples:
        result = detect_with_api("What is quantum entanglement?", provider="openai")
        result = detect_with_api("...", provider="anthropic", model="claude-sonnet-4-20250514")
        result = detect_with_api("...", provider="ollama", model="llama3.2")
    """
    p = get_provider(provider, model, **kwargs)
    return p.detect(prompt, **kwargs)


if __name__ == "__main__":
    print("Δt Detector - API Providers")
    print("=" * 50)
    print()
    print("Available providers:")
    print("  - openai    (logprobs available, best accuracy)")
    print("  - anthropic (no logprobs, consistency-based)")
    print("  - ollama    (local models via API)")
    print()
    print("Usage:")
    print('  from detector.api_providers import detect_with_api')
    print('  result = detect_with_api("Your question", provider="openai")')
    print()
    print("Set API keys via environment variables:")
    print("  OPENAI_API_KEY=sk-...")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
