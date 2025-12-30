#!/usr/bin/env python3
"""
MATS 10.0: Cross-Task Validation
==================================

Test if intervention breaks non-reasoning tasks.

Tasks:
1. Factual Recall (10 questions): "What is the capital of France?"
2. Translation (10 questions): "Translate to Spanish: Hello"
3. Summarization (5 examples): "Summarize this paragraph: ..."

Conditions:
1. Baseline: No intervention
2. Extreme: L23H5 at α=10.0 (the collapse value)

Validation:
- Non-reasoning tasks should remain accurate even at α=10.0
- This proves intervention is reasoning-specific, not global damage

Run:
    python cross_task_validation.py
"""

import sys
import re
from pathlib import Path
from typing import List, Dict
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from mats.model import load_model
from mats.hooks import add_scaling_hooks, reset_hooks
from mats.utils import set_seed, print_separator

set_seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# NON-REASONING TASKS
# ============================================================================

FACTUAL_QUESTIONS = [
    {"prompt": "What is the capital of France?\nAnswer:", "expected": "Paris"},
    {"prompt": "Who wrote 'The Iliad'?\nAnswer:", "expected": "Homer"},
    {"prompt": "What is the chemical symbol for gold?\nAnswer:", "expected": "Au"},
    {"prompt": "What planet is known as the Red Planet?\nAnswer:", "expected": "Mars"},
    {"prompt": "Who painted the Mona Lisa?\nAnswer:", "expected": "Leonardo da Vinci"},
    {"prompt": "What is the largest ocean on Earth?\nAnswer:", "expected": "Pacific"},
    {"prompt": "What year did World War II end?\nAnswer:", "expected": "1945"},
    {"prompt": "What is the capital of Japan?\nAnswer:", "expected": "Tokyo"},
    {"prompt": "Who discovered penicillin?\nAnswer:", "expected": "Fleming"},
    {"prompt": "What is the speed of light in km/s (approximately)?\nAnswer:", "expected": "300000"},
]

TRANSLATION_QUESTIONS = [
    {"prompt": "Translate to Spanish: Hello\nTranslation:", "expected": "Hola"},
    {"prompt": "Translate to French: Thank you\nTranslation:", "expected": "Merci"},
    {"prompt": "Translate to German: Good morning\nTranslation:", "expected": "Guten Morgen"},
    {"prompt": "Translate to Italian: I love you\nTranslation:", "expected": "Ti amo"},
    {"prompt": "Translate to Spanish: Goodbye\nTranslation:", "expected": "Adiós"},
    {"prompt": "Translate to French: Please\nTranslation:", "expected": "S'il vous plaît"},
    {"prompt": "Translate to German: Yes\nTranslation:", "expected": "Ja"},
    {"prompt": "Translate to Italian: No\nTranslation:", "expected": "No"},
    {"prompt": "Translate to Spanish: Water\nTranslation:", "expected": "Agua"},
    {"prompt": "Translate to French: Dog\nTranslation:", "expected": "Chien"},
]

SUMMARIZATION_EXAMPLES = [
    {
        "prompt": """Summarize in one sentence: The Amazon rainforest, often referred to as the 'lungs of the Earth,' produces about 20% of the world's oxygen. It spans across nine countries and is home to millions of species, many of which are still undiscovered. Deforestation poses a significant threat to this vital ecosystem.
Summary:""",
        "keywords": ["Amazon", "rainforest", "oxygen", "deforestation"],
    },
    {
        "prompt": """Summarize in one sentence: Climate change is causing global temperatures to rise, leading to melting ice caps, rising sea levels, and more frequent extreme weather events. Scientists warn that immediate action is needed to prevent catastrophic consequences.
Summary:""",
        "keywords": ["climate", "temperature", "rising", "action"],
    },
    {
        "prompt": """Summarize in one sentence: The invention of the printing press by Johannes Gutenberg in the 15th century revolutionized the spread of information, making books more accessible and contributing to the Renaissance and Reformation movements.
Summary:""",
        "keywords": ["printing press", "Gutenberg", "books", "information"],
    },
    {
        "prompt": """Summarize in one sentence: Artificial intelligence is transforming industries from healthcare to transportation, enabling faster diagnoses, autonomous vehicles, and personalized recommendations. However, concerns about job displacement and ethical use remain.
Summary:""",
        "keywords": ["AI", "artificial intelligence", "transforming", "industries"],
    },
    {
        "prompt": """Summarize in one sentence: The Great Wall of China, built over several centuries, stretches approximately 21,000 kilometers and was constructed to protect against invasions from northern tribes.
Summary:""",
        "keywords": ["Great Wall", "China", "kilometers", "protection"],
    },
]

# Math problem for comparison
MATH_QUESTIONS = [
    {"prompt": "Question: What is 15 + 27?\nAnswer:", "expected": "42"},
    {"prompt": "Question: What is 8 × 7?\nAnswer:", "expected": "56"},
    {"prompt": "Question: What is 100 - 37?\nAnswer:", "expected": "63"},
    {"prompt": "Question: What is 144 ÷ 12?\nAnswer:", "expected": "12"},
    {"prompt": "Question: What is 25% of 80?\nAnswer:", "expected": "20"},
]


def check_factual(output: str, expected: str) -> bool:
    """Check if output contains expected answer."""
    return expected.lower() in output.lower()


def check_translation(output: str, expected: str) -> bool:
    """Check if translation is correct (case insensitive)."""
    # Be lenient - check if expected word appears
    return expected.lower() in output.lower()


def check_summary(output: str, keywords: List[str]) -> bool:
    """Check if summary contains key concepts."""
    output_lower = output.lower()
    matches = sum(1 for kw in keywords if kw.lower() in output_lower)
    return matches >= len(keywords) // 2  # At least half the keywords


def check_math(output: str, expected: str) -> bool:
    """Check if math answer is correct."""
    numbers = re.findall(r'\b(\d+)\b', output)
    return expected in numbers


def check_coherence(output: str) -> bool:
    """Check if output is coherent."""
    if len(output) < 5:
        return False
    letter_ratio = sum(c.isalpha() for c in output) / max(1, len(output))
    return letter_ratio > 0.3


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print_separator("MATS 10.0: Cross-Task Validation")
    print("Testing if intervention breaks non-reasoning tasks\n")
    
    output_dir = Path("results/cross_task")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model("Qwen/Qwen2.5-7B-Instruct", device="cuda")
    
    TARGET_LAYER = 23
    TARGET_HEAD = 5
    EXTREME_ALPHA = 10.0
    
    conditions = [
        {"name": "Baseline", "alpha": 1.0},
        {"name": f"Extreme (α={EXTREME_ALPHA})", "alpha": EXTREME_ALPHA},
    ]
    
    results = []
    
    # Test each task type
    task_types = [
        ("Factual", FACTUAL_QUESTIONS, check_factual),
        ("Translation", TRANSLATION_QUESTIONS, check_translation),
        ("Math", MATH_QUESTIONS, check_math),
    ]
    
    for task_name, questions, check_fn in task_types:
        print(f"\n{task_name} Task ({len(questions)} questions)")
        
        for cond in conditions:
            n_correct = 0
            n_coherent = 0
            
            for q in tqdm(questions, desc=f"{task_name} - {cond['name']}", leave=False):
                reset_hooks(model)
                if cond['alpha'] != 1.0:
                    add_scaling_hooks(model, TARGET_LAYER, TARGET_HEAD, cond['alpha'])
                
                try:
                    output = model.generate(q['prompt'], max_new_tokens=50, temperature=0.0, do_sample=False)
                except Exception as e:
                    output = f"ERROR: {e}"
                finally:
                    reset_hooks(model)
                
                # Check correctness
                is_correct = check_fn(output, q['expected'])
                is_coherent = check_coherence(output)
                
                if is_correct:
                    n_correct += 1
                if is_coherent:
                    n_coherent += 1
            
            accuracy = n_correct / len(questions)
            coherence = n_coherent / len(questions)
            
            results.append({
                "task_type": task_name,
                "condition": cond['name'],
                "alpha": cond['alpha'],
                "accuracy": accuracy,
                "coherence": coherence,
                "n_correct": n_correct,
                "n_total": len(questions),
            })
            
            print(f"  {cond['name']}: {accuracy:.1%} accuracy, {coherence:.1%} coherent")
    
    # Special handling for summarization (keyword-based)
    print(f"\nSummarization Task ({len(SUMMARIZATION_EXAMPLES)} examples)")
    for cond in conditions:
        n_correct = 0
        n_coherent = 0
        
        for ex in tqdm(SUMMARIZATION_EXAMPLES, desc=f"Summarization - {cond['name']}", leave=False):
            reset_hooks(model)
            if cond['alpha'] != 1.0:
                add_scaling_hooks(model, TARGET_LAYER, TARGET_HEAD, cond['alpha'])
            
            try:
                output = model.generate(ex['prompt'], max_new_tokens=100, temperature=0.0, do_sample=False)
            except Exception as e:
                output = f"ERROR: {e}"
            finally:
                reset_hooks(model)
            
            is_correct = check_summary(output, ex['keywords'])
            is_coherent = check_coherence(output)
            
            if is_correct:
                n_correct += 1
            if is_coherent:
                n_coherent += 1
        
        accuracy = n_correct / len(SUMMARIZATION_EXAMPLES)
        coherence = n_coherent / len(SUMMARIZATION_EXAMPLES)
        
        results.append({
            "task_type": "Summarization",
            "condition": cond['name'],
            "alpha": cond['alpha'],
            "accuracy": accuracy,
            "coherence": coherence,
            "n_correct": n_correct,
            "n_total": len(SUMMARIZATION_EXAMPLES),
        })
        
        print(f"  {cond['name']}: {accuracy:.1%} accuracy, {coherence:.1%} coherent")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "cross_task_results.csv", index=False)
    print(f"\n✓ Saved: cross_task_results.csv")
    
    # Generate visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot for grouped bar chart
    pivot = df.pivot(index='task_type', columns='condition', values='accuracy')
    
    x = np.arange(len(pivot.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pivot['Baseline'], width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot[f'Extreme (α={EXTREME_ALPHA})'], width, 
                   label=f'Extreme (α={EXTREME_ALPHA})', color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.0%}', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.0%}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_title(f'Cross-Task Validation: Does α={EXTREME_ALPHA} Break Non-Reasoning Tasks?', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "cross_task_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: cross_task_comparison.png")
    
    # Summary
    print_separator("Summary")
    
    # Check if non-reasoning tasks are preserved
    non_reasoning = df[df['task_type'].isin(['Factual', 'Translation', 'Summarization'])]
    baseline_acc = non_reasoning[non_reasoning['condition'] == 'Baseline']['accuracy'].mean()
    extreme_acc = non_reasoning[non_reasoning['condition'] == f'Extreme (α={EXTREME_ALPHA})']['accuracy'].mean()
    
    math_baseline = df[(df['task_type'] == 'Math') & (df['condition'] == 'Baseline')]['accuracy'].values[0]
    math_extreme = df[(df['task_type'] == 'Math') & (df['condition'] == f'Extreme (α={EXTREME_ALPHA})')]['accuracy'].values[0]
    
    print(f"Non-Reasoning Tasks:")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"  Extreme α={EXTREME_ALPHA}: {extreme_acc:.1%}")
    print(f"  Degradation: {(baseline_acc - extreme_acc):.1%}")
    
    print(f"\nMath Task (for comparison):")
    print(f"  Baseline: {math_baseline:.1%}")
    print(f"  Extreme: {math_extreme:.1%}")
    
    is_task_specific = extreme_acc > 0.7 and abs(extreme_acc - baseline_acc) < 0.2
    print(f"\nIntervention is task-specific: {'✅ CONFIRMED' if is_task_specific else '❌ NOT CONFIRMED'}")
    
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
