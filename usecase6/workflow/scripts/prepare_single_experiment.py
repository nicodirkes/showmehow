import hashlib, sys, yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
outdir   = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(cfg_path.open())
if "experiment_hash" not in cfg:
    cfg["experiment_hash"] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:8]

out_path = outdir / f"experiment_{cfg['experiment_hash']}.yml"
yaml.dump(cfg, out_path.open("w"), default_flow_style=False, sort_keys=False)
