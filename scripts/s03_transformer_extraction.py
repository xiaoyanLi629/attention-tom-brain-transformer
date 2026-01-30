#!/usr/bin/env python3
"""
=============================================================================
Stage 3: Transformer Activation Extraction - Deep Mechanism Analysis
=============================================================================

Extract and analyze internal mechanisms of Transformer models on ToM tasks.

Deep Analyses:
    1. Attention Pattern Extraction
       - Layer-wise attention weights
       - Head-level attention specialization
       - Attention sparsity (entropy)
       
    2. Hidden State Extraction
       - Layer-wise representations
       - Token-level embeddings
       
    3. Attention Head Specialization
       - Which heads attend to mental state words
       - Social cue detection patterns
       
    4. Information Flow Analysis
       - Attention flow from social cues to predictions
       - Critical pathway identification

Models:
    - GPT-2-XL (1.5B) - smaller, faster
    - LLaMA-2-7B (7B) - larger, more capable
    - Mistral-7B (7B) - efficient attention

Outputs:
    - attention_weights/{model}/*.pkl
    - hidden_states/{model}/*.pkl
    - head_specialization.csv
    - attention_sparsity.csv
    - attention_flow.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure directories are initialized (for subprocess runs)
config.ensure_run_directories()

logger = config.setup_logging('transformer_extraction')

# =============================================================================
# SOCIAL COGNITION PROMPTS
# =============================================================================

# Theory of Mind scenarios
TOM_PROMPTS = {
    'mental_intentional': [
        {
            'text': "Sarah placed her keys on the kitchen counter. While she was in the bathroom, her husband moved the keys to the drawer. When Sarah comes back, she will look for her keys in the",
            'answer': "counter",
            'mental_words': ['placed', 'moved', 'look', 'will'],
            'agent_words': ['Sarah', 'husband', 'she'],
            'type': 'false_belief',
        },
        {
            'text': "John wants to surprise his wife with a birthday cake. He hides it in the garage. His wife, not knowing about the surprise, decides to clean the garage. When John goes to get the cake, he believes the cake is",
            'answer': "in the garage",
            'mental_words': ['wants', 'surprise', 'hides', 'knowing', 'believes'],
            'agent_words': ['John', 'wife', 'He', 'his'],
            'type': 'false_belief',
        },
        {
            'text': "Mary thinks that Tom is angry at her, but actually Tom is just tired. If someone asks Mary about Tom's feelings, Mary would say Tom is",
            'answer': "angry",
            'mental_words': ['thinks', 'angry', 'tired', 'feelings', 'say'],
            'agent_words': ['Mary', 'Tom', 'her', 'someone'],
            'type': 'mental_state',
        },
        {
            'text': "The child sees a chocolate on the table and wants to eat it. Her mother says 'No, that's for later.' The child feels",
            'answer': "disappointed",
            'mental_words': ['sees', 'wants', 'says', 'feels'],
            'agent_words': ['child', 'mother', 'Her', 'she'],
            'type': 'emotion_inference',
        },
    ],
    'random_control': [
        {
            'text': "The temperature today is 25 degrees Celsius. According to the weather forecast, tomorrow will be",
            'answer': "warmer/cooler",
            'mental_words': [],
            'agent_words': [],
            'type': 'factual',
        },
        {
            'text': "Water boils at 100 degrees Celsius at sea level. At higher altitudes, water boils at",
            'answer': "lower temperatures",
            'mental_words': [],
            'agent_words': [],
            'type': 'factual',
        },
        {
            'text': "The capital of France is Paris. The Eiffel Tower is located in",
            'answer': "Paris",
            'mental_words': [],
            'agent_words': [],
            'type': 'factual',
        },
    ],
}

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_transformer_model(model_config):
    """Load a Transformer model with attention output enabled"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = model_config['name']
    short_name = model_config['short_name']
    
    logger.info(f"Loading {short_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=config.MODEL_CACHE_DIR,
        )
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            
            if 'gpt2' in model_name.lower():
                # GPT-2 is small, load normally
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=config.MODEL_CACHE_DIR,
                    output_attentions=True,
                    output_hidden_states=True,
                ).cuda()
            elif gpu_memory < 20:
                # Use quantization for larger models
                logger.info("Using 8-bit quantization...")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=config.MODEL_CACHE_DIR,
                    output_attentions=True,
                    output_hidden_states=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=config.MODEL_CACHE_DIR,
                    output_attentions=True,
                    output_hidden_states=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            logger.warning("No GPU, loading on CPU")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=config.MODEL_CACHE_DIR,
                output_attentions=True,
                output_hidden_states=True,
            )
        
        model.eval()
        logger.info(f"✅ {short_name} loaded successfully")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load {short_name}: {e}")
        return None, None


# =============================================================================
# ATTENTION EXTRACTION
# =============================================================================

def extract_attention_patterns(model, tokenizer, prompt_info):
    """Extract attention patterns for a given prompt"""
    
    text = prompt_info['text']
    mental_words = prompt_info.get('mental_words', [])
    agent_words = prompt_info.get('agent_words', [])
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    # Extract attention weights
    # Shape: (n_layers, batch, n_heads, seq_len, seq_len)
    attentions = outputs.attentions
    attention_weights = torch.stack([attn.squeeze(0) for attn in attentions])  # (n_layers, n_heads, seq_len, seq_len)
    
    # Extract hidden states
    # Shape: (n_layers+1, batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states
    hidden_stack = torch.stack([h.squeeze(0) for h in hidden_states])  # (n_layers+1, seq_len, hidden_dim)
    
    # Get token information
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Identify mental state and agent tokens
    mental_token_ids = []
    agent_token_ids = []
    
    for i, token in enumerate(tokens):
        token_clean = token.replace('Ġ', '').replace('▁', '').lower()
        for word in mental_words:
            if word.lower() in token_clean or token_clean in word.lower():
                mental_token_ids.append(i)
                break
        for word in agent_words:
            if word.lower() in token_clean or token_clean in word.lower():
                agent_token_ids.append(i)
                break
    
    result = {
        'attention_weights': attention_weights.cpu().numpy(),  # (n_layers, n_heads, seq_len, seq_len)
        'hidden_states': hidden_stack.cpu().numpy(),  # (n_layers+1, seq_len, hidden_dim)
        'tokens': tokens,
        'text': text,
        'prompt_type': prompt_info['type'],
        'mental_token_ids': mental_token_ids,
        'agent_token_ids': agent_token_ids,
        'n_layers': len(attentions),
        'n_heads': attentions[0].shape[1],
    }
    
    return result


def compute_attention_sparsity(attention_weights):
    """
    Compute sparsity of attention distribution.
    
    Metrics:
        - Entropy: Lower = more focused attention
        - Top-k concentration: Higher = sparser
        - Effective rank: Lower = more specialized
    """
    # attention_weights: (n_layers, n_heads, seq_len, seq_len)
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    results = []
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attn = attention_weights[layer_idx, head_idx]  # (seq_len, seq_len)
            
            # Average across query positions
            attn_avg = attn.mean(axis=0)  # (seq_len,)
            
            # Entropy (normalized)
            epsilon = 1e-10
            entropy = -np.sum(attn_avg * np.log2(attn_avg + epsilon))
            max_entropy = np.log2(seq_len)
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Top-5 concentration
            top5_idx = np.argsort(attn_avg)[-5:]
            top5_concentration = attn_avg[top5_idx].sum()
            
            # Gini coefficient
            sorted_attn = np.sort(attn_avg)
            n = len(sorted_attn)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_attn))) / (n * np.sum(sorted_attn) + epsilon) - (n+1)/n
            
            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'entropy': float(norm_entropy),
                'top5_concentration': float(top5_concentration),
                'gini': float(gini),
                'sparsity': float(1 - norm_entropy),  # Inverted entropy
            })
    
    return results


def compute_head_specialization(attention_result):
    """
    Compute specialization of attention heads for mental state processing.
    
    A head is "ToM-specialized" if it attends more to mental state words
    than to other words.
    """
    attention_weights = attention_result['attention_weights']
    mental_token_ids = attention_result['mental_token_ids']
    agent_token_ids = attention_result['agent_token_ids']
    tokens = attention_result['tokens']
    
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    results = []
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attn = attention_weights[layer_idx, head_idx]  # (seq_len, seq_len)
            
            # Average attention received by each token
            attn_received = attn.mean(axis=0)  # (seq_len,)
            
            # Attention to mental tokens
            if len(mental_token_ids) > 0:
                mental_attn = attn_received[mental_token_ids].mean()
            else:
                mental_attn = 0.0
            
            # Attention to agent tokens
            if len(agent_token_ids) > 0:
                agent_attn = attn_received[agent_token_ids].mean()
            else:
                agent_attn = 0.0
            
            # Attention to other tokens
            other_ids = [i for i in range(seq_len) 
                        if i not in mental_token_ids and i not in agent_token_ids]
            if len(other_ids) > 0:
                other_attn = attn_received[other_ids].mean()
            else:
                other_attn = attn_received.mean()
            
            # Specialization scores
            mental_spec = mental_attn / (other_attn + 1e-10)
            agent_spec = agent_attn / (other_attn + 1e-10)
            
            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'mental_attention': float(mental_attn),
                'agent_attention': float(agent_attn),
                'other_attention': float(other_attn),
                'mental_specialization': float(mental_spec),
                'agent_specialization': float(agent_spec),
                'is_tom_head': mental_spec > 1.5,  # Threshold for specialization
            })
    
    return results


def compute_attention_flow(attention_result):
    """
    Compute information flow through attention layers.
    
    Traces how attention flows from input tokens to the final prediction.
    Identifies critical pathways for social information.
    """
    attention_weights = attention_result['attention_weights']
    mental_token_ids = attention_result['mental_token_ids']
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    # Average attention across heads
    avg_attn = attention_weights.mean(axis=1)  # (n_layers, seq_len, seq_len)
    
    # Compute cumulative attention flow (rollout)
    # How much attention flows from input to final position through layers
    rollout = np.eye(seq_len)
    
    layer_flows = []
    for layer_idx in range(n_layers):
        attn = avg_attn[layer_idx]
        rollout = np.matmul(attn, rollout)
        
        # Attention from final token to mental tokens
        if len(mental_token_ids) > 0:
            final_to_mental = rollout[-1, mental_token_ids].mean()
        else:
            final_to_mental = 0.0
        
        layer_flows.append({
            'layer': layer_idx,
            'final_to_mental_flow': float(final_to_mental),
            'mean_rollout': float(rollout[-1].mean()),
            'max_rollout': float(rollout[-1].max()),
        })
    
    return {
        'layer_flows': layer_flows,
        'total_mental_flow': float(sum(lf['final_to_mental_flow'] for lf in layer_flows)),
        'peak_mental_layer': max(layer_flows, key=lambda x: x['final_to_mental_flow'])['layer'],
    }


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def run_transformer_extraction():
    """Run complete Transformer extraction pipeline"""
    
    logger.info("="*60)
    logger.info("Stage 3: Transformer Activation Extraction")
    logger.info("="*60)
    
    # Create output directories
    attention_dir = config.TRANSFORMER_ATTENTION_DIR / "attention_weights"
    hidden_dir = config.TRANSFORMER_ATTENTION_DIR / "hidden_states"
    attention_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    
    all_sparsity_data = []
    all_specialization_data = []
    all_flow_data = []
    
    # Process each model
    for model_config in config.TRANSFORMER_CONFIG['models']:
        model_name = model_config['short_name']
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {model_name}")
        logger.info(f"{'='*40}")
        
        model, tokenizer = load_transformer_model(model_config)
        
        if model is None:
            logger.warning(f"Skipping {model_name} - could not load")
            continue
        
        # Create model-specific directories
        model_attn_dir = attention_dir / model_name.replace(' ', '_').replace('-', '_')
        model_hidden_dir = hidden_dir / model_name.replace(' ', '_').replace('-', '_')
        model_attn_dir.mkdir(exist_ok=True)
        model_hidden_dir.mkdir(exist_ok=True)
        
        # Process all prompts
        all_prompts = []
        for category, prompts in TOM_PROMPTS.items():
            for prompt in prompts:
                prompt['category'] = category
                all_prompts.append(prompt)
        
        for i, prompt_info in enumerate(tqdm(all_prompts, desc=f"Extracting {model_name}")):
            
            try:
                # Extract attention patterns
                result = extract_attention_patterns(model, tokenizer, prompt_info)
                result['model'] = model_name
                result['prompt_idx'] = i
                result['category'] = prompt_info['category']
                
                # Save attention weights
                attn_path = model_attn_dir / f"prompt_{i}.pkl"
                with open(attn_path, 'wb') as f:
                    # Only save necessary info (attention weights are large)
                    save_result = {k: v for k, v in result.items() if k != 'hidden_states'}
                    pickle.dump(save_result, f)
                
                # Save hidden states (separate due to size)
                hidden_path = model_hidden_dir / f"prompt_{i}.pkl"
                with open(hidden_path, 'wb') as f:
                    pickle.dump({
                        'hidden_states': result['hidden_states'],
                        'tokens': result['tokens'],
                        'prompt_idx': i,
                    }, f)
                
                # Compute sparsity
                sparsity_results = compute_attention_sparsity(result['attention_weights'])
                for sp in sparsity_results:
                    sp['model'] = model_name
                    sp['prompt_idx'] = i
                    sp['category'] = prompt_info['category']
                    sp['prompt_type'] = prompt_info['type']
                all_sparsity_data.extend(sparsity_results)
                
                # Compute head specialization
                spec_results = compute_head_specialization(result)
                for spec in spec_results:
                    spec['model'] = model_name
                    spec['prompt_idx'] = i
                    spec['category'] = prompt_info['category']
                    spec['prompt_type'] = prompt_info['type']
                all_specialization_data.extend(spec_results)
                
                # Compute attention flow
                flow_result = compute_attention_flow(result)
                flow_result['model'] = model_name
                flow_result['prompt_idx'] = i
                flow_result['category'] = prompt_info['category']
                flow_result['prompt_type'] = prompt_info['type']
                all_flow_data.append(flow_result)
                
            except Exception as e:
                logger.warning(f"Error processing prompt {i}: {e}")
                continue
        
        # Clean up model to free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save aggregate results
    sparsity_df = pd.DataFrame(all_sparsity_data)
    sparsity_df.to_csv(config.TRANSFORMER_ATTENTION_DIR / "attention_sparsity.csv", index=False)
    
    specialization_df = pd.DataFrame(all_specialization_data)
    specialization_df.to_csv(config.TRANSFORMER_ATTENTION_DIR / "head_specialization.csv", index=False)
    
    with open(config.TRANSFORMER_ATTENTION_DIR / "attention_flow.json", 'w') as f:
        json.dump(all_flow_data, f, indent=2)
    
    logger.info(f"\nSaved sparsity data: {len(sparsity_df)} records")
    logger.info(f"Saved specialization data: {len(specialization_df)} records")
    logger.info(f"Saved flow data: {len(all_flow_data)} records")
    
    # Compute summary statistics
    stats_results = compute_extraction_stats(sparsity_df, specialization_df, all_flow_data)
    
    with open(config.TRANSFORMER_ATTENTION_DIR / "extraction_stats.json", 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    # Print summary
    print_extraction_summary(stats_results)
    
    return sparsity_df, specialization_df, all_flow_data


def compute_extraction_stats(sparsity_df, specialization_df, flow_data):
    """Compute summary statistics from extraction results"""
    
    stats = {
        'models': {},
        'overall': {},
    }
    
    for model in sparsity_df['model'].unique():
        model_sparsity = sparsity_df[sparsity_df['model'] == model]
        model_spec = specialization_df[specialization_df['model'] == model]
        
        # Sparsity by category
        tom_sparsity = model_sparsity[model_sparsity['category'] == 'mental_intentional']['sparsity'].mean()
        ctrl_sparsity = model_sparsity[model_sparsity['category'] == 'random_control']['sparsity'].mean()
        
        # ToM-specialized heads
        n_tom_heads = model_spec[model_spec['is_tom_head']].groupby(['prompt_idx', 'layer']).size().mean()
        
        stats['models'][model] = {
            'mean_sparsity': float(model_sparsity['sparsity'].mean()),
            'mean_entropy': float(model_sparsity['entropy'].mean()),
            'tom_sparsity': float(tom_sparsity) if not np.isnan(tom_sparsity) else 0.0,
            'control_sparsity': float(ctrl_sparsity) if not np.isnan(ctrl_sparsity) else 0.0,
            'mean_tom_heads_per_layer': float(n_tom_heads) if not np.isnan(n_tom_heads) else 0.0,
            'mean_mental_specialization': float(model_spec['mental_specialization'].mean()),
            'n_layers': int(model_spec['layer'].max() + 1),
            'n_heads': int(model_spec['head'].max() + 1),
        }
    
    # Overall statistics
    if len(sparsity_df) > 0:
        stats['overall'] = {
            'mean_sparsity': float(sparsity_df['sparsity'].mean()),
            'sparsity_std': float(sparsity_df['sparsity'].std()),
            'mean_entropy': float(sparsity_df['entropy'].mean()),
            'tom_vs_control_sparsity_diff': float(
                sparsity_df[sparsity_df['category'] == 'mental_intentional']['sparsity'].mean() -
                sparsity_df[sparsity_df['category'] == 'random_control']['sparsity'].mean()
            ) if 'random_control' in sparsity_df['category'].values else 0.0,
        }
    
    return stats


def print_extraction_summary(stats):
    """Print summary of extraction results"""
    
    logger.info("\n" + "="*60)
    logger.info("Transformer Extraction Summary")
    logger.info("="*60)
    
    for model, model_stats in stats['models'].items():
        logger.info(f"\n{model}:")
        logger.info(f"  Layers: {model_stats['n_layers']}, Heads: {model_stats['n_heads']}")
        logger.info(f"  Mean sparsity: {model_stats['mean_sparsity']:.3f}")
        logger.info(f"  Mean entropy: {model_stats['mean_entropy']:.3f}")
        logger.info(f"  ToM sparsity: {model_stats['tom_sparsity']:.3f}")
        logger.info(f"  Control sparsity: {model_stats['control_sparsity']:.3f}")
        logger.info(f"  ToM-specialized heads/layer: {model_stats['mean_tom_heads_per_layer']:.1f}")
        logger.info(f"  Mental state specialization: {model_stats['mean_mental_specialization']:.2f}x")
    
    if stats['overall']:
        logger.info(f"\nOverall:")
        logger.info(f"  Mean sparsity: {stats['overall']['mean_sparsity']:.3f} ± {stats['overall']['sparsity_std']:.3f}")
        logger.info(f"  ToM vs Control sparsity diff: {stats['overall']['tom_vs_control_sparsity_diff']:.3f}")


if __name__ == "__main__":
    sparsity_df, spec_df, flow_data = run_transformer_extraction()
    print("\n✅ Stage 3 completed: Transformer extraction")

