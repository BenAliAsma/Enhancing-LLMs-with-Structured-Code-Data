#!/usr/bin/env python3
"""
PatchGenerator that handles model instantiation and robust patch generation
"""

import time
import logging
from typing import Dict, Any, Tuple
from src.llm_models import BaseLLMModel

logger = logging.getLogger(__name__)

class PatchGenerator:
    """Handles patch generation using different models and strategies"""
    
    def __init__(self):
        pass  # Removed unused self.loaded_models cache
    
    def generate_patch(
        self, 
        model: BaseLLMModel, 
        context: str, 
        strategy: str, 
        model_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a patch using the specified model, context, and strategy.
        
        Args:
            model: The model instance to use
            context: The context string for generation
            strategy: The patch generation strategy
            model_name: The model's name (for logs/metrics)
            
        Returns:
            Tuple of (patch string, metrics dictionary)
        """
        try:
            # Ensure model is loaded
            if not self._is_model_loaded(model):
                logger.info(f"Loading model: {model_name}")
                model.load_model()
            
            # Generate the patch using the model
            patch, metrics = model.generate_patch(context, strategy)
            
            # Add tracking metadata
            metrics.update({
                'model_name': model_name,
                'strategy': strategy,
                'timestamp': time.time(),
                'success': True
            })
            
            return patch, metrics
        
        except Exception as e:
            logger.exception(f"[ERROR] Failed to generate patch using {model_name} with strategy '{strategy}'")
            
            # Return a safe fallback result
            fallback_patch = self._create_fallback_patch(model_name, strategy, str(e))
            fallback_metrics = {
                'model_name': model_name,
                'strategy': strategy,
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'fallback': True
            }
            return fallback_patch, fallback_metrics

    def _is_model_loaded(self, model: BaseLLMModel) -> bool:
        """Check if the model has been properly loaded"""
        return hasattr(model, 'model') and model.model is not None

    def _create_fallback_patch(self, model_name: str, strategy: str, error: str) -> str:
        """Create a fallback patch message when generation fails"""
        return f"""# Patch generation failed
# Model: {model_name}
# Strategy: {strategy}
# Error: {error}

# Placeholder patch:
# - Ensure the model is properly configured and reachable.
# - Check API keys or model file paths.
# - Retry after fixing the above issues.

# This block simulates the output when actual patch generation fails.
"""

    def get_model_info(self, model: BaseLLMModel) -> Dict[str, Any]:
        """Retrieve metadata about the model"""
        if hasattr(model, 'get_model_info'):
            return model.get_model_info()
        
        return {
            'name': getattr(model, 'model_name', 'unknown'),
            'type': getattr(getattr(model, 'config', {}), 'get', lambda x, y='unknown': y)('type'),
            'loaded': self._is_model_loaded(model)
        }
