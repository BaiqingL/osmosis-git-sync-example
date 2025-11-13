import re
import json
from typing import Any, Dict, Optional
from difflib import SequenceMatcher
from osmosis_ai import osmosis_reward

def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last valid JSON object from text."""
    # Find all potential JSON blocks (between curly braces)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = list(re.finditer(json_pattern, text, re.DOTALL))
    
    # Try to parse from the last match backwards
    for match in reversed(matches):
        try:
            json_str = match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON found in simple matches, try to find the last { and parse until valid
    # This handles multi-line JSON better
    brace_positions = [i for i, c in enumerate(text) if c == '{']
    for start_pos in reversed(brace_positions):
        for end_pos in range(len(text), start_pos, -1):
            try:
                candidate = text[start_pos:end_pos]
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    
    return None


def normalize_value(value: Any) -> Optional[str]:
    """Normalize a value to handle null/None/empty cases."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ['none', 'null', '', 'n/a', 'na']:
            return None
        return value.lower()
    if isinstance(value, (int, float)):
        return str(value).lower()
    return str(value).lower()


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two text strings.
    Uses word-level overlap to handle partial matches.
    Returns a score between 0.0 and 1.0.
    """
    if text1 == text2:
        return 1.0
    
    # Tokenize into words (split on whitespace and common separators)
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        # If either is empty, use character-level similarity
        return SequenceMatcher(None, text1, text2).ratio()
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Also consider character-level similarity for cases like minor typos
    char_similarity = SequenceMatcher(None, text1, text2).ratio()
    
    # Return weighted average (favor word overlap for longer texts)
    if len(words1) > 3 or len(words2) > 3:
        return 0.7 * jaccard + 0.3 * char_similarity
    else:
        return 0.5 * jaccard + 0.5 * char_similarity


def compare_values(model_value: Any, gt_value: Any, key: str = "") -> float:
    """
    Compare two values and return a score between 0.0 and 1.0.
    
    Args:
        model_value: Value from model output
        gt_value: Ground truth value
        key: The key name (for context, if needed)
    
    Returns:
        Score between 0.0 and 1.0
    """
    norm_model = normalize_value(model_value)
    norm_gt = normalize_value(gt_value)
    
    # Both are null/empty - perfect match
    if norm_model is None and norm_gt is None:
        return 1.0
    
    # GT is null but model has a value - this is GOOD! Model extracted more info.
    # Give full credit (or high partial credit) for extracting additional data
    if norm_gt is None and norm_model is not None:
        return 1.0  # Full credit for extracting extra information
    
    # Model is null but GT has a value - this is BAD! Model missed required info.
    if norm_model is None and norm_gt is not None:
        return 0.0
    
    # Exact match (after normalization)
    if norm_model == norm_gt:
        return 1.0
    
    # Try to parse as numbers for numeric comparison
    try:
        num_model = float(norm_model.replace(',', '').replace('-', ''))
        num_gt = float(norm_gt.replace(',', '').replace('-', ''))
        
        # Check if numbers are close (within 0.1% or absolute difference < 0.01)
        if abs(num_model - num_gt) < 0.01:
            return 1.0
        elif abs(num_model - num_gt) / max(abs(num_gt), 1e-10) < 0.001:
            return 1.0
        else:
            # Partial credit for numbers that are somewhat close
            rel_diff = abs(num_model - num_gt) / max(abs(num_gt), 1e-10)
            if rel_diff < 0.1:  # Within 10%
                return 0.5
            return 0.0
    except (ValueError, AttributeError):
        # Not numbers, treat as text
        pass
    
    # Text similarity for partial matches
    similarity = compute_text_similarity(norm_model, norm_gt)
    
    # Apply threshold - give partial credit for high similarity
    if similarity >= 0.9:
        return 1.0
    elif similarity >= 0.7:
        return 0.8
    elif similarity >= 0.5:
        return 0.5
    elif similarity >= 0.3:
        return 0.2
    else:
        return 0.0


@osmosis_reward
def json_extraction_reward(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """
    Reward function for JSON extraction tasks.
    
    Compares the extracted JSON from model output with ground truth JSON.
    - Handles null/None/empty values appropriately
    - Uses fuzzy text matching for partial matches
    - Doesn't penalize for extra keys in model output
    - Gives partial credit for close matches
    
    Args:
        solution_str: Model output containing JSON
        ground_truth: Ground truth JSON string
        extra_info: Additional information (unused)
        **kwargs: Additional arguments
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract JSON from model output
    model_json = extract_last_json(solution_str)
    if model_json is None:
        # No valid JSON found in model output
        return 0.0
    
    # Parse ground truth JSON
    try:
        if isinstance(ground_truth, str):
            gt_json = json.loads(ground_truth)
        else:
            gt_json = ground_truth
    except (json.JSONDecodeError, TypeError):
        # Invalid ground truth format
        return 0.0
    
    # Normalize keys (case-insensitive comparison)
    model_json_normalized = {k.lower(): v for k, v in model_json.items()}
    gt_json_normalized = {k.lower(): v for k, v in gt_json.items()}
    
    # Calculate score for each key in ground truth
    scores = []
    for gt_key, gt_value in gt_json_normalized.items():
        if gt_key in model_json_normalized:
            model_value = model_json_normalized[gt_key]
            score = compare_values(model_value, gt_value, key=gt_key)
            scores.append(score)
        else:
            # Key missing in model output
            scores.append(0.0)
    
    # Return average score across all ground truth keys
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)