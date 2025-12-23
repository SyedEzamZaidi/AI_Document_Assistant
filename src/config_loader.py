"""
Configuration loader for AI Document Assistant.
Reads settings from YAML files and provides helper functions.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_config(config_file: str = "config/models.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_file (str): Path to YAML config file
        
    Returns:
        Dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


# ================================================
# EMBEDDING MODEL FUNCTIONS
# ================================================

def get_embedding_model_name(config: Dict) -> str:
    """
    Get the full name of current embedding model from config.
    Handles both local and cloud providers.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        str: Full model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    """
    current = config['embeddings']['current']
    provider = config['embeddings']['current_provider']
    
    # Choose the right model list based on provider
    if provider == 'local':
        models = config['embeddings']['local_models']
    else:
        models = config['embeddings']['cloud_models']
    
    # Find the model
    for model in models:
        if model['name'] == current:
            return model['full_name']
    
    raise ValueError(f"Embedding model '{current}' not found in {provider} models")


def get_all_embedding_models(config: Dict, provider: str = 'local') -> List[Dict]:
    """
    Get all available embedding models for a provider.
    
    Args:
        config (Dict): Configuration dictionary
        provider (str): 'local' or 'cloud'
        
    Returns:
        List[Dict]: List of model dictionaries
    """
    if provider == 'local':
        return config['embeddings']['local_models']
    else:
        return config['embeddings']['cloud_models']


# ================================================
# LLM FUNCTIONS
# ================================================

def get_llm_model_name(config: Dict) -> str:
    """
    Get the current LLM model name from config.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        str: LLM model name (e.g., "llama3.1")
    """
    return config['llm']['current']


def get_llm_provider(config: Dict) -> str:
    """
    Get the current LLM provider.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        str: Provider name ('local' or 'cloud')
    """
    return config['llm']['current_provider']


def get_all_llm_models(config: Dict, provider: str = 'local') -> List[Dict]:
    """
    Get all available LLM models for a provider.
    
    Args:
        config (Dict): Configuration dictionary
        provider (str): 'local' or 'cloud'
        
    Returns:
        List[Dict]: List of model dictionaries
    """
    if provider == 'local':
        return config['llm']['local_models']
    else:
        return config['llm']['cloud_models']


# ================================================
# CHUNKING FUNCTIONS
# ================================================

def get_chunking_method(config: Dict) -> str:
    """
    Get the current chunking method class name from config.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        str: Class name (e.g., "RecursiveCharacterTextSplitter")
    """
    current = config['chunking']['current_method']
    methods = config['chunking']['available_methods']
    
    for method in methods:
        if method['name'] == current:
            return method['class']
    
    raise ValueError(f"Chunking method '{current}' not found in config")


def get_chunking_params(config: Dict) -> Tuple[int, int]:
    """
    Get current chunking parameters from config.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple[int, int]: (chunk_size, chunk_overlap)
    """
    current_params = config['chunking']['current_params']
    param_sets = config['chunking']['parameter_sets']
    
    for param_set in param_sets:
        if param_set['name'] == current_params:
            return param_set['chunk_size'], param_set['chunk_overlap']
    
    raise ValueError(f"Parameter set '{current_params}' not found in config")


def get_all_chunking_param_sets(config: Dict) -> List[Dict]:
    """
    Get all available chunking parameter sets for testing.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        List[Dict]: List of parameter set dictionaries
    """
    return config['chunking']['parameter_sets']


# ================================================
# JUDGE MODEL FUNCTIONS
# ================================================

def get_judge_model_name(config: Dict) -> str:
    """
    Get the current AI judge model name from config.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        str: Judge model name (e.g., "llama3.1")
    """
    return config['evaluation']['current_judge']


def get_judge_provider(config: Dict) -> str:
    """
    Get the current judge provider.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        str: Provider name ('local' or 'cloud')
    """
    return config['evaluation']['judge_provider']


def get_all_judge_models(config: Dict, provider: str = 'local') -> List[Dict]:
    """
    Get all available judge models for a provider.
    
    Args:
        config (Dict): Configuration dictionary
        provider (str): 'local' or 'cloud'
        
    Returns:
        List[Dict]: List of judge model dictionaries
    """
    if provider == 'local':
        return config['evaluation']['local_judge_models']
    else:
        return config['evaluation']['cloud_judge_models']


# ================================================
# VECTOR STORE FUNCTIONS
# ================================================

def get_vector_store_config(config: Dict) -> Dict:
    """
    Get vector store configuration.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict: Vector store configuration
    """
    return config['vector_store']


def get_retrieval_params(config: Dict) -> Dict:
    """
    Get retrieval parameters.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict: Retrieval parameters (top_k, min_similarity_score)
    """
    return config['retrieval']


# ================================================
# TESTING FUNCTIONS
# ================================================

def is_multi_model_test_enabled(config: Dict) -> bool:
    """
    Check if multi-model testing is enabled.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        bool: True if multi-model testing is enabled
    """
    return config['testing']['multi_model_test']


def get_models_to_test(config: Dict) -> Dict[str, List[str]]:
    """
    Get lists of models to test in comparison mode.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict: Dictionary with 'embeddings', 'llms', 'chunking_params' lists
    """
    testing = config['testing']
    
    return {
        'local_embeddings': testing.get('local_embeddings_to_test', []),
        'local_llms': testing.get('local_llms_to_test', []),
        'cloud_embeddings': testing.get('cloud_embeddings_to_test', []),
        'cloud_llms': testing.get('cloud_llms_to_test', []),
        'chunking_params': testing.get('chunking_params_to_test', [])
    }


# ================================================
# DEPLOYMENT FUNCTIONS
# ================================================

def get_deployment_config(config: Dict, scenario: str) -> Dict:
    """
    Get deployment configuration for a specific scenario.
    
    Args:
        config (Dict): Configuration dictionary
        scenario (str): 'development', 'production_budget', 
                       'production_quality', or 'enterprise_compliance'
        
    Returns:
        Dict: Deployment configuration for the scenario
    """
    if scenario not in config['deployment']:
        raise ValueError(f"Unknown deployment scenario: {scenario}")
    
    return config['deployment'][scenario]


# ================================================
# TEST BLOCK
# ================================================

if __name__ == "__main__":
    # Test the config loader
    print("="*70)
    print("CONFIG LOADER TEST")
    print("="*70)
    
    config = load_config()
    print("✅ Config loaded successfully!\n")
    
    # Embeddings
    print("="*70)
    print("EMBEDDINGS CONFIG")
    print("="*70)
    print(f"Current: {config['embeddings']['current']}")
    print(f"Provider: {config['embeddings']['current_provider']}")
    print(f"Full name: {get_embedding_model_name(config)}")
    print(f"\nLocal models available: {len(get_all_embedding_models(config, 'local'))}")
    print(f"Cloud models available: {len(get_all_embedding_models(config, 'cloud'))}\n")
    
    # LLMs
    print("="*70)
    print("LLM CONFIG")
    print("="*70)
    print(f"Current: {get_llm_model_name(config)}")
    print(f"Provider: {get_llm_provider(config)}")
    print(f"\nLocal models available: {len(get_all_llm_models(config, 'local'))}")
    print(f"Cloud models available: {len(get_all_llm_models(config, 'cloud'))}\n")
    
    # Chunking
    print("="*70)
    print("CHUNKING CONFIG")
    print("="*70)
    print(f"Method: {get_chunking_method(config)}")
    chunk_size, chunk_overlap = get_chunking_params(config)
    print(f"Parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"\nAvailable parameter sets:")
    for param_set in get_all_chunking_param_sets(config):
        print(f"  - {param_set['name']}: "
              f"size={param_set['chunk_size']}, "
              f"overlap={param_set['chunk_overlap']} "
              f"({param_set['description']})\n")
    
    # Judge
    print("="*70)
    print("AI JUDGE CONFIG")
    print("="*70)
    print(f"Current judge: {get_judge_model_name(config)}")
    print(f"Provider: {get_judge_provider(config)}")
    print(f"\nLocal judge models: {len(get_all_judge_models(config, 'local'))}")
    print(f"Cloud judge models: {len(get_all_judge_models(config, 'cloud'))}\n")
    
    # Vector Store
    print("="*70)
    print("VECTOR STORE CONFIG")
    print("="*70)
    vs_config = get_vector_store_config(config)
    print(f"Type: {vs_config['type']}")
    print(f"Provider: {vs_config['provider']}")
    print(f"Persist directory: {vs_config['persist_directory']}")
    retrieval = get_retrieval_params(config)
    print(f"\nRetrieval: top_k={retrieval['top_k']}, "
          f"min_score={retrieval['min_similarity_score']}\n")
    
    # Testing
    print("="*70)
    print("TESTING CONFIG")
    print("="*70)
    print(f"Multi-model test enabled: {is_multi_model_test_enabled(config)}")
    if is_multi_model_test_enabled(config):
        models_to_test = get_models_to_test(config)
        print(f"\nModels to test:")
        print(f"  Local embeddings: {models_to_test['local_embeddings']}")
        print(f"  Local LLMs: {models_to_test['local_llms']}")
        print(f"  Chunking params: {models_to_test['chunking_params']}\n")
    
    # Deployment scenarios
    print("="*70)
    print("DEPLOYMENT SCENARIOS")
    print("="*70)
    scenarios = ['development', 'production_budget', 'production_quality', 'enterprise_compliance']
    for scenario in scenarios:
        dep_config = get_deployment_config(config, scenario)
        print(f"\n{scenario.replace('_', ' ').title()}:")
        print(f"  Mode: {dep_config['mode']}")
        print(f"  Embedding: {dep_config['embedding']}")
        print(f"  LLM: {dep_config['llm']}")
        print(f"  Judge: {dep_config['judge']}")
        print(f"  Reason: {dep_config['reason']}")
    
    print("\n" + "="*70)
    print("✅ ALL CONFIG FUNCTIONS WORKING!")
    print("="*70)