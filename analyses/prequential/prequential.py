from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from g_metrology.data import load_yaml
from g_metrology.prequential import prequential_world
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results'/'machine_readable'; OUT.mkdir(parents=True,exist_ok=True)
D=load_yaml(ROOT/'data/codata_2018_g.yaml')
r=prequential_world(D,start_train=7,draws=1400,burn=900,chains=2)
(OUT/'prequential_results.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
