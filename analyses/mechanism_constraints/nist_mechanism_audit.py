from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm

ROOT=Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / 'src'))
from g_metrology.nist_repro import (
    NIST_VALUES, reported_covariance,
    TABLE14_AUTO, TABLE15_TORQUE, TABLE17_BUDGET, NIST_THERMAL, NIST_FINAL_G
)
from g_metrology.torque import fractional_torque_fit
OUT=ROOT/'results'/'machine_readable'; OUT.mkdir(parents=True, exist_ok=True)

# Table 15: selected raw torque differences and Type-A uncertainties.
_keys=('copper_0','copper_120','copper_240','sapphire')
d=np.array([(TABLE15_TORQUE[k]['free']-TABLE15_TORQUE[k]['servo'])*1e3 for k in _keys])
s=np.array([math.hypot(TABLE15_TORQUE[k]['free_u_typeA'],TABLE15_TORQUE[k]['servo_u_typeA'])*1e3 for k in _keys])
w=1/s**2
common=float((w@d)/(w.sum())); common_se=float(1/math.sqrt(w.sum()))
q=float(np.sum(((d-common)/s)**2)); p=float(chi2.sf(q,3))

G=NIST_VALUES.copy(); C=reported_covariance()
Lcu=np.array([-1,1,0,0.]); Lsa=np.array([0,0,-1,1.])
contrasts=[]
for lab,L in [('copper',Lcu),('sapphire',Lsa)]:
    delta=float(L@G); se=float(math.sqrt(L@C@L)); mean=float(np.mean(G[np.flatnonzero(L)])); z=delta/se
    contrasts.append({'label':lab,'delta_G_e15':delta/1e-15,'se_G_e15':se/1e-15,'relative_ppm':delta/mean*1e6,'z':z,'p_two_sided':float(2*norm.sf(abs(z)))})

# Convert common torque offset to apparent G shift.
Nfree_nm=np.array([TABLE15_TORQUE[k]['free'] for k in _keys])
Gref=NIST_FINAL_G
pred=np.array([Gref*(common*1e-12)/(n*1e-9) for n in Nfree_nm])
pred_se=np.array([Gref*(common_se*1e-12)/(n*1e-9) for n in Nfree_nm])

# Table 14 pairwise support intervals, using the published endpoints.
aut_pair={}
for lab,sv,fr in [('cu0','copper_servo_0','copper_free_0'),('cu120','copper_servo_120','copper_free_120'),('cu240','copper_servo_240','copper_free_240'),('sapphire','sapphire_servo','sapphire_free')]:
    a1,b1=TABLE14_AUTO[fr]['a'],TABLE14_AUTO[fr]['b']; a0,b0=TABLE14_AUTO[sv]['a'],TABLE14_AUTO[sv]['b']
    aut_pair[lab]={'min_ppm':a1-b0,'max_ppm':b1-a0,'width_ppm':(b1-a0)-(a1-b0)}

# Table 17 relative budget -> absolute pN m using the free torque signal for each configuration.
labels=TABLE17_BUDGET['labels']; rows=TABLE17_BUDGET['rows']; abs_torque_scales={}
for k,name in zip(_keys,['copper_servo','copper_free','sapphire_servo','sapphire_free']):
    signal_nm=TABLE15_TORQUE[k]['free']
    vals=rows[name]
    abs_torque_scales[name]={lab:float(v*1e-6*signal_nm) for lab,v in zip(labels,vals)}

thermal_pNm=float(NIST_THERMAL['torque_scale_pNm'])
required_ppm={'copper':contrasts[0]['relative_ppm'],'sapphire':contrasts[1]['relative_ppm']}
# Illustrative discriminator predictions used in manuscript Table 8.
# The absolute-offset hypothesis predicts Delta G/G = common / N_grav.
# The fractional-scale hypothesis predicts a constant fitted f.
f_hat, _, _, _ = fractional_torque_fit()

discriminator_torque_levels_nNm=np.array([14.0,31.0,62.0])
discriminator_absolute_ppm=common*1000.0/discriminator_torque_levels_nNm
discriminator_fractional_ppm=np.full(
    len(discriminator_torque_levels_nNm),
    1e6*f_hat,
)

out={
 'source':'Schlamminger et al. 2026, Metrologia 63 025012, Tables 14-19 and Sections 3, 7-9',
 'table15_common_mode_offset':{'estimate_pNm':common,'se_pNm':common_se,'chi2':q,'dof':3,'p':p},
 'common_offset_predicted_G_shift_e15':(pred/1e-15).tolist(),
 'common_offset_predicted_G_shift_se_e15':(pred_se/1e-15).tolist(),
 'discriminator_predictions':{
   'gravitational_torque_levels_nNm':discriminator_torque_levels_nNm.tolist(),
   'absolute_offset_predicted_ppm':discriminator_absolute_ppm.tolist(),
   'fractional_scale_predicted_ppm':discriminator_fractional_ppm.tolist(),
 },
 'published_G_mode_contrasts':contrasts,
 'required_mode_difference_ppm':required_ppm,
 'autocollimator_table14_pair_support_ppm':aut_pair,
 'thermal_torque_scale_pNm_from_paper_statement':thermal_pNm,
 'table17_absolute_torque_uncertainty_scales_pNm':abs_torque_scales,
 'interpretation':{
   'common_torque':'All four raw Table-15 free-minus-servo torque differences are consistent with one common absolute torque offset using Type-A errors alone.',
   'material':'The larger sapphire G contrast can arise from a common absolute torque offset because the sapphire gravitational torque is smaller; this does not establish a material-dependent gravitational effect.',
   'autocollimator':'The very broad Table-14 correction support for sapphire-free is large enough to overlap the observed sapphire mode contrast, but copper pairwise support is much smaller than the observed copper contrast. Because the calibration errors are fully correlated, this interval check cannot identify causation.',
   'thermal':'The paper reports a gas-mediated thermal torque scale of about 10^-4 (0.01%) of the lowest gravitational signal, about 1.4 pN m, which is comparable to the observed ~1.9 pN m mode offset. The paper also reports no significant time-dependent torque after servo-voltage application, so this is a magnitude comparison, not a mechanism identification.',
   'electrostatic':'Capacitance and voltage uncertainties are each 6 ppm in servo mode; the paper gives no signed residual capable of converting these uncertainty magnitudes into a predicted 2 pN m bias.'
 }
}
(OUT/'nist_mechanism_constraints.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
