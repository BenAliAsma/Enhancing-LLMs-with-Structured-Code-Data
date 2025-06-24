import os
import json
import requests

# Data paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
LOGS_DIR = os.path.join(BASE_DIR, '..', 'logs')
MATCHED_BLOCKS_PATH = os.path.join(DATA_DIR, 'matched_blocks_ranked.json')
SNIPPETS_PATH = os.path.join(DATA_DIR, 'outputs', 'all_snippets_consolidated.py')

# Dataset configuration
DATASET_CONFIG = {
    'url': 'https://datasets-server.huggingface.co/rows',
    'dataset': 'princeton-nlp/SWE-bench_Verified',
    'config': 'default',
    'split': 'test',
    'offset': 0,
    'length': 100
}

def load_dataset_problems():
    """Load problems from SWE-bench dataset. Returns a list of problem configurations."""
    dataset_path = os.path.join(DATA_DIR, "swe_bench_problems.json")
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            return json.load(f)

    try:
        response = requests.get(DATASET_CONFIG['url'], params={
            'dataset': DATASET_CONFIG['dataset'],
            'config': DATASET_CONFIG['config'],
            'split': DATASET_CONFIG['split'],
            'offset': DATASET_CONFIG['offset'],
            'length': DATASET_CONFIG['length']
        })
        response.raise_for_status()
        data = response.json()

        problems = [
            {
                'instance_id': row['row'].get('instance_id'),
                'repo': row['row'].get('repo'),
                'base_commit': row['row'].get('base_commit'),
                'problem_statement': row['row'].get('problem_statement'),
                'patch': row['row'].get('patch'),
                'test_patch': row['row'].get('test_patch'),
                'version': row['row'].get('version'),
                'created_at': row['row'].get('created_at'),
            }
            for row in data.get('rows', [])
        ]

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(dataset_path, 'w') as f:
            json.dump(problems, f, indent=2)

        return problems

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def get_problem_by_index(index=0):
    problems = load_dataset_problems()
    if 0 <= index < len(problems):
        problem = problems[index]
        return {
            'commit': problem['base_commit'],
            'repo_name': problem['repo'],
            'problem_stmt': problem['problem_statement'],
            'test_patch': problem['test_patch'],
            'instance_id': problem['instance_id'],
            'created_at': problem['created_at'],
            'version': problem['version']
        }
    return None

def get_problem_by_id(instance_id):
    problems = load_dataset_problems()
    for problem in problems:
        if problem['instance_id'] == instance_id:
            return {
                'commit': problem['base_commit'],
                'repo_name': problem['repo'],
                'problem_stmt': problem['problem_statement'],
                'test_patch': problem['test_patch'],
                'instance_id': problem['instance_id'],
                'created_at': problem['created_at'],
                'version': problem['version']
            }
    return None

def get_all_problems():
    problems = load_dataset_problems()
    return [
        {
            'commit': problem['base_commit'],
            'repo_name': problem['repo'],
            'problem_stmt': problem['problem_statement'],
            'test_patch': problem['test_patch'],
            'instance_id': problem['instance_id'],
            'created_at': problem['created_at'],
            'version': problem['version']
        }
        for problem in problems
    ]

# Backward-compatible config with default values
problem_config = get_problem_by_index(0)
if problem_config:
    commit = problem_config['commit']
    repo_name = problem_config['repo_name']
    problem_stmt = problem_config['problem_stmt']
    test_patch = problem_config['test_patch']
    date = problem_config['created_at']
    version = problem_config['version']
else:
    # Default values if no problem is found
    commit = None
    repo_name = None
    problem_stmt = None
    test_patch = None
    date = None
    version = None

# Context strategies
STRATEGIES = ['minimal', 'balanced', 'comprehensive', 'rag_style']

# Model configurations
# Updated MODEL_CONFIGS section for config.py
MODEL_CONFIGS = {
     'qwen2.5_coder_7B_0.3_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.3,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_coder_7B_0.7_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.7,
        'max_length': 131072,
        'max_new_tokens': 32768
    },

 'qwen2.5_coder_1.5B_0.3_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.3,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_coder_1.5B_0.7_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.7,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_coder_3B_0.3_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-3B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.3,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_coder_3B_0.7_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-3B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.7,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_coder_14B_0.3_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-14B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.3,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_coder_14B_0.7_temp': {
        'model_path': 'Qwen/Qwen2.5-Coder-14B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.7,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2_72B_0.3_temp': {
        'model_path': 'Qwen/Qwen2-72B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.3,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2_72B_0.7_temp': {
        'model_path': 'Qwen/Qwen2-72B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.7,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_72B_0.3_temp': {
        'model_path': 'Qwen/Qwen2.5-72B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.3,
        'max_length': 131072,
        'max_new_tokens': 32768
    },
    'qwen2.5_72B_0.7_temp': {
        'model_path': 'Qwen/Qwen2.5-72B-Instruct',
        'type': 'causal_lm',
        'max_tokens': 131072,
        'temperature': 0.7,
        'max_length': 131072,
        'max_new_tokens': 32768
    }



}