import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, Tuple
import time
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLMModel(ABC):
    """Base class for all LLM models"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model_name = model_config['name']
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def generate_patch(self, context: str, strategy: str) -> Tuple[str, Dict[str, Any]]:
        """Generate a patch given context and strategy"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'type': self.config.get('type', 'unknown'),
            'device': str(self.device),
            'loaded': self.model is not None
        }

class HuggingFaceLLM(BaseLLMModel):
    """HuggingFace transformer models"""
    
    def load_model(self):
        """Load HuggingFace model"""
        try:
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
                
            logger.info(f"✓ Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {self.model_name}: {str(e)}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a smaller fallback model"""
        fallback_name = "microsoft/DialoGPT-small"
        try:
            logger.info(f"Loading fallback model: {fallback_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_name)
            self.model = self.model.to(self.device)
            self.model_name = fallback_name
            logger.info("✓ Fallback model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load fallback model: {str(e)}")
            raise
    
    def generate_patch(self, context: str, strategy: str) -> Tuple[str, Dict[str, Any]]:
        """Generate patch using HuggingFace model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Create prompt based on strategy
        prompt = self._create_prompt(context, strategy)
        
        # Tokenize input
        start_time = time.time()
        
        try:
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=2048, 
                truncation=True
            ).to(self.device)
            
            input_length = inputs.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.config.get('max_tokens', 512),
                    do_sample=True,
                    temperature=self.config.get('temperature', 0.7),
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            generation_time = time.time() - start_time
            output_length = outputs.shape[1] - input_length
            
            metrics = {
                'model': self.model_name,
                'strategy': strategy,
                'input_tokens': input_length,
                'output_tokens': output_length,
                'generation_time': generation_time,
                'tokens_per_second': output_length / generation_time if generation_time > 0 else 0,
                'context_length': len(context),
                'success': True
            }
            
            return response, metrics
            
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {str(e)}")
            # Return a mock patch for demonstration
            mock_response = self._create_mock_patch(strategy)
            metrics = {
                'model': self.model_name,
                'strategy': strategy,
                'input_tokens': 0,
                'output_tokens': len(mock_response.split()),
                'generation_time': time.time() - start_time,
                'tokens_per_second': 0,
                'context_length': len(context),
                'success': False,
                'error': str(e)
            }
            return mock_response, metrics
    
    def _create_prompt(self, context: str, strategy: str) -> str:
        """Create prompt based on strategy"""
        strategy_instructions = {
            'minimal': "Provide a concise, targeted fix focusing only on the core issue.",
            'balanced': "Consider the broader context while maintaining focus on the main issue.",
            'comprehensive': "Provide a thorough analysis considering all aspects of the codebase.",
            'rag_style': "Base your solution primarily on the provided code snippets."
        }
        
        instruction = strategy_instructions.get(strategy, "Generate an appropriate patch.")
        
        return f"""You are an expert software engineer. Analyze the following code issue and generate a patch in git diff format.

{context}

Strategy: {strategy}
Instructions: {instruction}

Generate a patch in the following format:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line_number,lines_count +line_number,lines_count @@
 context_line
-removed_line
+added_line
 context_line
```

Patch:"""
    
    def _create_mock_patch(self, strategy: str) -> str:
        """Create a mock patch for fallback"""
        return f"""```
```
# Generated using {strategy} strategy
"""

class MockClaudeModel(BaseLLMModel):
    """Mock Claude model for demonstration"""
    
    def load_model(self):
        """Mock load for Claude"""
        logger.info("Mock Claude model loaded (API-based)")
        self.model = "mock_claude"
        self.tokenizer = "mock_tokenizer"
    
    def generate_patch(self, context: str, strategy: str) -> Tuple[str, Dict[str, Any]]:
        """Mock generation for Claude"""
        time.sleep(1)  # Simulate API call
        
        
        metrics = {
            'model': self.model_name,
            'strategy': strategy,
            'input_tokens': len(context.split()) // 4,  # Rough estimate
            'output_tokens': len(response.split()),
            'generation_time': 1.0,
            'tokens_per_second': len(response.split()),
            'context_length': len(context),
            'success': True
        }
        
        return response, metrics

def create_model(model_name: str, config: Dict[str, Any]) -> BaseLLMModel:
    if config['type'] == 'api' or 'claude' in model_name.lower():
        return MockClaudeModel(config)
    else:
        return HuggingFaceLLM(config)

