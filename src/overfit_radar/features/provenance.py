"""Content-addressed dataset provenance and mandatory training input gate."""
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from overfit_radar.config import get_project_root, load_yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root=None):
    root = root or get_project_root()
    def inventory(folder, pattern):
        return {str(p.relative_to(root)): digest(p) for p in sorted((root/folder).rglob(pattern))}
    configs = inventory('configs','*.yaml')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    return dict(commit=commit, commit_status='available' if commit else 'no_git_repository',
                config_hash=hashlib.sha256(json.dumps(configs,sort_keys=True).encode()).hexdigest(),
                configs=configs, source_hashes=inventory('src','*.py'), data_hashes=inventory('data/raw','*.parquet'),
                cost_schedule_version=load_yaml(root/'configs/protocol.yaml')['cost_schedule_version'])


def save_versioned(table, directory):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'training.parquet'
    if path.exists():
        raise FileExistsError('Version already exists; choose a new version directory')
    table.to_parquet(path,index=False)
    manifest=fingerprint()
    manifest.update(dataset_sha256=digest(path), rows=len(table), columns=list(table.columns),
                    data_snapshot_date=datetime.now(timezone.utc).isoformat(),
                    snapshot_date_meaning='Local cache snapshot captured now; original download timestamp unavailable')
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    return path


def load_training_table(path):
    """Refuse stale data, code, config or table bytes; Git HEAD is provenance only."""
    path=Path(path); manifest=json.loads(path.with_suffix('.manifest.json').read_text())
    current=fingerprint()
    for key,value in current.items():
        if key in {'commit', 'commit_status'}:
            continue  # A documentation-only commit does not change research inputs.
        if manifest.get(key)!=value:
            raise ValueError(f'Stale or mismatched dataset: {key}')
    if digest(path)!=manifest['dataset_sha256']:
        raise ValueError('Dataset content hash mismatch')
    return pd.read_parquet(path)


def load_for_model_training(path):
    """Training entry point: require current data and completed baseline artifacts."""
    path=Path(path)
    table=load_training_table(path)
    gates=json.loads((path.parent/'gates.json').read_text())
    if gates.get('gates_1_to_4')!='passed' or not gates.get('baseline_results_computed') or gates.get('dataset_sha256')!=digest(path):
        raise ValueError('Training gates have not passed for this dataset')
    for name,expected in gates['baseline_hashes'].items():
        if digest(path.parent/name)!=expected:
            raise ValueError('Baseline artifact mismatch')
    return table
