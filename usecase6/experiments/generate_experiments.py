#!/usr/bin/env python3
"""
Generate parameter combinations from experiments.yml
Creates experiment_<hash>.yml files for each combination
"""

import yaml
import hashlib
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List


def generate_hash(params_dict: Dict[str, Any]) -> str:
    """Generate a short hash from parameter dict"""
    params_str = '|'.join(
        f"{k}={v}" for k, v in sorted(
            [(k, str(v)) for k, v in params_dict.items()],
            key=lambda x: x[0]
        )
    )
    return hashlib.sha256(params_str.encode()).hexdigest()[:8]


def deep_merge(base: Dict, updates: Dict) -> Dict:
    """Recursively merge updates into base dict"""
    result = base.copy()
    for key, value in updates.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_iterations(iterations: Dict, prefix: str = "") -> Dict:
    """Flatten nested iteration parameters using dot notation"""
    flat = {}

    for key, value in iterations.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_iterations(value, new_key))
        else:
            flat[new_key] = value

    return flat


def unflatten_dict(flat_dict: Dict) -> Dict:
    """Convert flat dict with dot notation back to nested dict"""
    result = {}

    for key, value in flat_dict.items():
        parts = key.split('.')
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result


def generate_combinations(experiment_config: Dict) -> List[Dict]:
    """Generate all parameter combinations from experiments config"""
    base_params = experiment_config.get('base_params', {})
    iteration_params = experiment_config.get('iterations', {})

    if not iteration_params:
        return [base_params]

    # Flatten nested iteration parameters to dot notation
    flat_iterations = flatten_iterations(iteration_params)

    # Get parameter names and their value lists
    param_names = sorted(flat_iterations.keys())
    value_lists = [flat_iterations[name] for name in param_names]

    # Generate cartesian product of all value combinations
    combinations = []
    for values in itertools.product(*value_lists):
        combo_dict = dict(zip(param_names, values))
        # Convert flat dict with dot notation to nested dict
        nested_combo = unflatten_dict(combo_dict)

        merged = deep_merge(base_params, nested_combo)
        combinations.append(merged)

    return combinations


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_experiments.py <experiments.yml> [output_dir]")
        sys.exit(1)

    experiments_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    if not experiments_file.exists():
        print(f"Error: {experiments_file} not found", file=sys.stderr)
        sys.exit(1)

    # Load experiments config
    with open(experiments_file) as f:
        config = yaml.safe_load(f)

    # Support both single-group (base_params + iterations) and multi-group formats
    if 'groups' in config:
        groups = config['groups']
    else:
        groups = [config]

    combinations = []
    for group in groups:
        group_combinations = generate_combinations(group)
        combinations.extend(group_combinations)
        print(f"Group '{group.get('name', 'default')}': {len(group_combinations)} combinations", file=sys.stderr)

    print(f"Generated {len(combinations)} parameter combinations total", file=sys.stderr)

    # Create output directory if not current directory
    if output_dir != Path("."):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write params files
    for params in combinations:
        hash_val = generate_hash(params)

        # Add experiment_hash to params
        params_with_hash = {**params, 'experiment_hash': hash_val}

        # Write to file
        output_file = output_dir / f"experiment_{hash_val}.yml"
        with open(output_file, 'w') as f:
            yaml.dump(params_with_hash, f, default_flow_style=False, sort_keys=False)

        print(f"Created {output_file}", file=sys.stderr)

    print(f"All params files written to {output_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()
