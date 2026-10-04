from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from g_metrology.nist_structure import fit_nist_structures
from g_metrology.data import load_yaml, active_covariance_from_dataset
cfg=load_yaml(ROOT/'data'/'nist2026_configurations.yaml')
y=cfg.y
C=active_covariance_from_dataset(cfg)
fits=fit_nist_structures(y,C)
rows=[]
for f in fits:
    rows.append({'name':f.name,'mu_si':f.mu_si,'components_si':f.components_si,'loglik':f.loglik,'aic':f.aic,'success':f.success})
rows=sorted(rows,key=lambda r:r['aic'])
gap_to_dark = next(r['aic'] for r in rows if r['name']=='independent_dark')-rows[0]['aic']
out={'input_source':'data/nist2026_configurations.yaml','fits':rows,'interpretation':{'lowest_aic_model':rows[0]['name'],'aic_gap_to_independent_dark':gap_to_dark,'selection_note':'AIC differences this small should be treated as effectively indistinguishable.'}}
(ROOT/'results'/'machine_readable'/'nist_structure_profiles.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
