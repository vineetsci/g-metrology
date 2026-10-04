from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from g_metrology.prequential_variance import run_prequential_variance
import json
root=Path(__file__).resolve().parents[2]
out=run_prequential_variance(root/'data/codata_2018_g.yaml')
(root/'results'/'machine_readable'/'prequential_variance_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if not isinstance(v,list)},indent=2))
for r in out['rows']:
    print(r['holdout_label'], {k:round(r[k],4) for k in ('free_lambda','free_dark','free_both')}, 'lambda',round(r['free_lambda_lambda'],3),'delta',round(r['free_dark_delta'],2))
