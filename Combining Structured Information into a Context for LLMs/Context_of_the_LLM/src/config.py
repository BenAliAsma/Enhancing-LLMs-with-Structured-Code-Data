import os

commit= "fa4e8d1cd279acf9b24560813c8652494ccd5922"
date="2023-02-06T21:56:51Z"
version="5.1"
repo_name="astropy/astropy"
problem_stmt = """
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
[False, True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True, True, False, False],
[ True, True, False, False],
[False, False, True, False],
[False, False, False, True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True, True, False, False],
[ True, True, False, False],
[False, False, True, True],
[False, False, True, True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?
"""

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MATCHED_BLOCKS_PATH = os.path.join(DATA_DIR, 'matched_blocks_ranked.json')
SNIPPETS_PATH = os.path.join(DATA_DIR, 'outputs', 'all_snippets_consolidated.py')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# Model configurations
MODEL_CONFIGS = {

    'deepseek': {
        'name': 'deepseek-ai/deepseek-coder-1.3b-instruct',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # Replace Claude with CodeLlama - excellent for code generation
    'codellama': {
        'name': 'codellama/CodeLlama-7b-Instruct-hf',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # Replace Llama7b with StarCoder - specifically trained for code
    'starcoder': {
        'name': 'bigcode/starcoder-1b',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # Replace Llama13b with CodeGen - Salesforce's code generation model
    'codegen': {
        'name': 'Salesforce/codegen-350M-mono',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # WizardCoder - very good for code generation
    'wizardcoder': {
        'name': 'WizardLM/WizardCoder-Python-7B-V1.0',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # Magicoder - Microsoft's code model
    'magicoder': {
        'name': 'ise-uiuc/Magicoder-S-DS-6.7B',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # CodeT5+ - encoder-decoder architecture, good for code understanding
    'codet5': {
        'name': 'Salesforce/codet5p-770m',
        'type': 'seq2seq',
        'max_tokens': 2048,
        'temperature': 0.7
    },

    # Smaller CodeGen
    'codegen_small': {
        'name': 'Salesforce/codegen-350M-mono',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # Small StarCoder
    'starcoder_small': {
        'name': 'bigcode/tiny_starcoder_py',
        'type': 'causal_lm',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    
    # CodeBERT for understanding + generation
    'codebert': {
        'name': 'microsoft/codebert-base',
        'type': 'masked_lm',  # Different type - needs special handling
        'max_tokens': 512,
        'temperature': 0.7
    },

    # Larger CodeLlama
    'codellama_large': {
        'name': 'codellama/CodeLlama-13b-Instruct-hf',
        'type': 'causal_lm',
        'max_tokens': 4096,
        'temperature': 0.7
    },
    
    # Larger StarCoder
    'starcoder_large': {
        'name': 'bigcode/starcoder-15.5b',
        'type': 'causal_lm',
        'max_tokens': 4096,
        'temperature': 0.7
    },
    
    # WizardCoder larger
    'wizardcoder_large': {
        'name': 'WizardLM/WizardCoder-15B-V1.0',
        'type': 'causal_lm',
        'max_tokens': 4096,
        'temperature': 0.7
    }
}

# Context strategies
STRATEGIES = ['minimal', 'balanced', 'comprehensive', 'rag_style']