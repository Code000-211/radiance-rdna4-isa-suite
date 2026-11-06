"""Nonnegative latency surrogate with shape-disjoint fit/calibration/test groups.

The calibration envelope uses the maximum log-error across candidates at each
calibration shape. It is finite-sample split-conformal in form, conditional on
exchangeable workload groups; deterministic benchmark grids do NOT establish that
assumption or a production guarantee. Held-out errors and selection regret are
reported even when disappointing. CPU evidence can never calibrate a GPU profile.
"""
from __future__ import annotations
import math
import random
import statistics
from .common import need, number, integer, digest, fields, hash_string
from .model import FEATURES, MODEL_VERSION, features, workload, candidate, profile, catalog_check, shape_key


def vector(f: dict) -> list[float]:
    return [f['features'][k] for k in FEATURES]


def nnls(x: list[list[float]], y: list[float], iterations: int = 4000) -> tuple[list[float],dict]:
    """Cyclic coordinate descent, nonnegative weighted least squares (relative loss).
    Column normalization keeps feature units from determining convergence.
    """
    need(len(x)==len(y) and len(x)>=8,'at least 8 training observations required')
    cols=len(FEATURES)
    need(all(len(r)==cols for r in x),'feature width mismatch')
    need(all(math.isfinite(t) and t>0 for t in y),'positive finite targets required')
    scales=[max(abs(r[j]) for r in x) or 1.0 for j in range(cols)]
    a=[[r[j]/scales[j]/t for j in range(cols)] for r,t in zip(x,y)]
    z=[0.0]*cols;pred=[0.0]*len(x);last=float('inf');converged=False
    for epoch in range(iterations):
        for j in range(cols):
            den=sum(r[j]*r[j] for r in a)
            if den==0:continue
            num=sum(r[j]*(1-pred[i]+r[j]*z[j]) for i,r in enumerate(a))
            value=max(0.,num/den);delta=value-z[j];z[j]=value
            for i,r in enumerate(a):pred[i]+=r[j]*delta
        loss=sum((v-1)**2 for v in pred)/len(pred)
        if abs(last-loss)<=1e-11*max(1.,last if math.isfinite(last) else 1.):
            converged=True;break
        last=loss
    return [z[j]/scales[j] for j in range(cols)],{'solver':'nonnegative coordinate descent; relative squared loss',
        'iterations':epoch+1,'converged':converged,'training_relative_mse':loss}


def dot(weights:list[float], x:list[float])->float:
    return max(1e-9,sum(a*b for a,b in zip(weights,x)))


def dataset_check(d: dict, cat: dict, p: dict) -> list[dict]:
    catalog_check(cat);profile(p)
    fields(d,{'schema','profile_sha256','records'}, {'notes'},'measurements')
    need(d['schema']=='radiance-planner-measurements/1','unsupported measurements')
    need(d['profile_sha256']==digest(p),'measurement/profile fingerprint differs')
    need(isinstance(d['records'],list) and len(d['records'])<=100000,'invalid measurement record array')
    table={c['id']:c for c in cat['candidates']};rows=[];seen=set();binary_by_candidate={}
    for r in d['records']:
        fields(r,{'workload','candidate_id','candidate_sha256','backend','arch','environment_sha256','metric',
                  'samples_us','correctness_status','evidence'},set(),'measurement')
        workload(r['workload']);need(r['candidate_id'] in table,'measurement candidate absent from catalog')
        c=table[r['candidate_id']]
        need(r['candidate_sha256']==digest(c),'candidate changed since measurement')
        need(r['backend']==p['backend']==c['backend'] and r['arch']==p['arch'],'CPU/GPU or architecture contamination')
        need(r['environment_sha256']==digest(p['environment']),'different environment/clock/compiler/measurement mode')
        need(r['metric']=='window_us','only comparable operation-window times accepted')
        need(r['correctness_status']=='RUNTIME_CHECKS_PASS','failed or unverified benchmark rejected')
        need(isinstance(r['samples_us'],list) and 3<=len(r['samples_us'])<=10000,'need 3..10000 retained samples')
        for t in r['samples_us']:number(t,'sample_us',positive=True)
        ev=r['evidence'];fields(ev,{'origin','report_sha256','fixture_sha256','worker_sha256','plugin_sha256'}, {'path'},'evidence')
        need(ev['origin']=='radreplay-verified-import','only checked replay imports accepted')
        for k in ('report_sha256','fixture_sha256','worker_sha256','plugin_sha256'):hash_string(ev[k],k)
        if c['identity']['kind']=='plugin':need(ev['plugin_sha256']==c['identity']['sha256'],'plugin identity mismatch')
        if p['backend']=='aql':
            need(c['identity'].get('plugin_sha256')==ev['plugin_sha256'],'GPU catalog needs an exact compiled plugin binding')
        old=binary_by_candidate.setdefault(c['id'],ev['plugin_sha256'])
        need(old==ev['plugin_sha256'],'mixed candidate plugin builds')
        key=shape_key(r['workload']);unique=(key,c['id'])
        need(unique not in seen,'duplicate shape/candidate; concatenate retained samples explicitly first')
        seen.add(unique)
        f=features(r['workload'],c,p)
        rows.append({'record':r,'group':key,'candidate':c,'features':f,'x':vector(f),'y':statistics.median(r['samples_us'])})
    need(bool(rows),'zero measurements')
    return rows


def split_groups(keys:list[str], *, seed:int, ncal:int, ntest:int)->dict:
    keys=sorted(set(keys));integer(ncal,'calibration groups');integer(ntest,'test groups')
    need(len(keys)-ncal-ntest>=8,'too few shape groups for training')
    random.Random(seed).shuffle(keys)
    return {'train':sorted(keys[ncal+ntest:]),'calibration':sorted(keys[:ncal]),'test':sorted(keys[ncal:ncal+ntest])}


def fit(d:dict,cat:dict,p:dict,*,seed:int=0,ncal:int=10,ntest:int=10,alpha:float=.1)->dict:
    number(alpha,'alpha',positive=True);need(alpha<.5,'alpha must be below 0.5')
    rows=dataset_check(d,cat,p)
    split=split_groups([r['group'] for r in rows],seed=seed,ncal=ncal,ntest=ntest)
    ids=sorted(c['id'] for c in cat['candidates'])
    # Complete panels avoid comparing winners against unmeasured alternatives.
    panels={k:{r['candidate']['id'] for r in rows if r['group']==k} for k in set(r['group'] for r in rows)}
    need(all(v==set(ids) for v in panels.values()),'incomplete candidate panel: no silently missing rivals')
    models={};train=set(split['train']);cal=set(split['calibration']);test=set(split['test'])
    for cid in ids:
        rs=[r for r in rows if r['candidate']['id']==cid and r['group'] in train]
        ws,solver=nnls([r['x'] for r in rs],[r['y'] for r in rs])
        formats=sorted({r['record']['workload']['contract'] for r in rs})
        domains={a:[min(r['record']['workload'][a] for r in rs),max(r['record']['workload'][a] for r in rs)] for a in ('M','N','K')}
        states=sorted({r['record']['workload']['cache_state'] for r in rs})
        models[cid]={'candidate_sha256':digest(rs[0]['candidate']),'coefficients':ws,'solver':solver,'domain':domains,
                     'contracts':formats,'cache_states':states,'training_rows':len(rs)}
    def predict(r):return dot(models[r['candidate']['id']]['coefficients'],r['x'])
    group_scores=[]
    for g in sorted(cal):
        group_scores.append(max(abs(math.log(r['y']/predict(r))) for r in rows if r['group']==g))
    quantile_rank=math.ceil((len(group_scores)+1)*(1-alpha))
    q=sorted(group_scores)[quantile_rank-1] if quantile_rank<=len(group_scores) else None
    test_rows=[];regrets=[];top1=0;top3=0;simultaneous=0;separated=0;separated_correct=0
    for g in sorted(test):
        rs=[r for r in rows if r['group']==g]
        order=sorted(rs,key=lambda r:(predict(r),r['candidate']['id']))
        best=min(rs,key=lambda r:r['y']);chosen=order[0]
        regret=chosen['y']/best['y']-1;regrets.append(regret)
        top1+=chosen['candidate']['id']==best['candidate']['id'];top3+=best in order[:3]
        covered=q is not None and all(abs(math.log(r['y']/predict(r)))<=q for r in rs)
        simultaneous+=covered
        is_sep=q is not None and len(order)>1 and predict(order[0])*math.exp(q)<min(predict(r)*math.exp(-q) for r in order[1:])
        separated+=is_sep;separated_correct+=is_sep and chosen['candidate']['id']==best['candidate']['id']
        for r in rs:
            yhat=predict(r);inside=all(models[r['candidate']['id']]['domain'][a][0]<=r['record']['workload'][a]<=models[r['candidate']['id']]['domain'][a][1] for a in ('M','N','K'))
            test_rows.append({'shape':{a:r['record']['workload'][a] for a in ('M','N','K')},'group':g,'candidate_id':r['candidate']['id'],
                'measured_us':r['y'],'predicted_us':yhat,'relative_error':abs(yhat/r['y']-1),'in_training_box':inside,
                'calibrated_envelope_us':None if q is None else [yhat*math.exp(-q),yhat*math.exp(q)]})
    errors=[r['relative_error'] for r in test_rows]
    result={'schema':'radiance-planner-calibration/1','model':MODEL_VERSION,'feature_order':list(FEATURES),
        'profile_sha256':digest(p),'backend':p['backend'],'arch':p['arch'],'environment_sha256':digest(p['environment']),
        'catalog_sha256':digest(cat),'measurement_sha256':digest(d),'seed':seed,'split':split,'models':models,
        'envelope':{'alpha':alpha,'group_log_error_scores':group_scores,'order_statistic':quantile_rank,'log_radius':q,
           'meaning':'candidate-simultaneous shape-group residual envelope; exchangeability required for nominal coverage',
           'production_coverage_guaranteed':False},
        'validation':{'test_groups':len(test),'test_rows':len(test_rows),'top1':top1/len(test),'top3':top3/len(test),
           'median_relative_latency_error':statistics.median(errors),'p90_relative_latency_error':sorted(errors)[math.ceil(.9*len(errors))-1],
           'median_selection_regret':statistics.median(regrets),'max_selection_regret':max(regrets),'per_shape_regret':regrets,
           'simultaneous_envelope_coverage':simultaneous/len(test),'separated_decisions':separated,'separated_correct':separated_correct},
        'held_out_predictions':test_rows,'gpu_calibration':p['backend']=='aql','deployment_approved':False,
        'notes':['Latency-surrogate coefficients are NOT physical bandwidth or instruction throughput.',
                 'Training-box membership alone does not establish interpolation accuracy.',
                 'Evidence JSON is trusted local input, not a remote attestation.','CPU results cannot be transferred to R9700.']}
    return result


def calibrated_prediction(cal:dict,w:dict,c:dict,p:dict,f:dict)->dict:
    need(cal.get('schema')=='radiance-planner-calibration/1' and cal.get('model')==MODEL_VERSION,'unsupported calibration/model')
    need(cal.get('profile_sha256')==digest(p),'calibration profile changed')
    need(cal.get('backend')==p['backend'] and cal.get('arch')==p['arch'],'calibration hardware mismatch')
    need(cal.get('environment_sha256')==digest(p['environment']),'calibration environment mismatch')
    need(cal.get('feature_order')==list(FEATURES),'calibration feature order changed')
    m=cal['models'].get(c['id'])
    if m is None or m['candidate_sha256']!=digest(c):return {'status':'NEW_OR_CHANGED_CANDIDATE','prediction_us':None,'interval_us':None}
    need(len(m['coefficients'])==len(FEATURES),'bad coefficient count')
    for coeff in m['coefficients']:number(coeff,'coefficient')
    y=dot(m['coefficients'],vector(f));reasons=[]
    for a in ('M','N','K'):
        if not m['domain'][a][0]<=w[a]<=m['domain'][a][1]:reasons.append(a+' outside training range')
    if w['contract'] not in m['contracts'] or w['cache_state'] not in m['cache_states']:reasons.append('contract/cache outside calibration')
    if f['cold_resource_floor_us'] is not None and y < f['cold_resource_floor_us']:
        reasons.append('prediction below declared cold resource floor; investigate model or evidence')
    q=cal['envelope']['log_radius']
    if q is not None:
        number(q,'log envelope radius');need(q<=50,'unstable/unbounded envelope')
    eligible=not reasons and q is not None
    return {'status':'CALIBRATED_IN_RANGE' if eligible else 'EXTRAPOLATION_OR_UNBOUNDED',
            'prediction_us':y,'interval_us':[y*math.exp(-q),y*math.exp(q)] if eligible else None,
            'reasons':reasons,'measured_here':False}
