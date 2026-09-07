# SPDX-License-Identifier: GPL-3.0-only
# Archived experiment harness; run base/traffic/planned/nm in that order to populate feature caches.
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from pipeline import ID,TARGET,TIME,departures,features,rmse,sha256,write_json
from traffic_features import enriched_features, add_planned_features, planned_baseline, add_nm_observed_features, nm_observed_baseline, nm_schedule_fallback_baseline

p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--features',choices=['base','traffic','planned','nm'],default='traffic')
p.add_argument('--iterations',type=int,default=1200)
p.add_argument('--depth',type=int,default=8)
p.add_argument('--cap',type=float,default=0)
p.add_argument('--validation',choices=['july','october','seasonal'],default='july')
p.add_argument('--refit',action='store_true')
p.add_argument('--device',choices=['CPU','GPU'],default='CPU')
p.add_argument('--residual',action='store_true')
p.add_argument('--schedule-fallback',action='store_true')
args=p.parse_args()
root=Path(__file__).resolve().parent/'runs'
out=root/args.name
assert not out.exists()
out.mkdir()
start=time.monotonic()
paths=sorted((root/'data').glob('training_*.parquet'))
raw=pd.concat([pd.read_parquet(f) for f in paths],ignore_index=True)
data=departures(raw)
y=data[TARGET].astype(float)
cache=root/('nm-features-v1.parquet' if args.features=='nm' else 'planned-features-v1.parquet' if args.features=='planned' else 'traffic-features-v1.parquet' if args.features=='traffic' else 'base-features.parquet')
if cache.exists():
    x=pd.read_parquet(cache)
    assert x.index.equals(data.index)
else:
    if args.features=='nm':
        x=add_nm_observed_features(pd.read_parquet(root/'planned-features-v1.parquet'),data)
    elif args.features=='planned':
        x=add_planned_features(pd.read_parquet(root/'traffic-features-v1.parquet'),data)
    else:
        x=enriched_features(data,raw) if args.features=='traffic' else features(data,'challenge_context')
    x.to_parquet(cache)
category=list(x.select_dtypes(include=['object','str']).columns)
clock=pd.to_datetime(data[TIME],utc=True)
validation=clock.ge('2025-07-01') if args.validation=='july' else clock.ge('2025-10-01') if args.validation=='october' else clock.dt.month.isin([1,7])
fit=~validation & y.ge(0)
offset=(nm_observed_baseline(x) if args.features=='nm' else planned_baseline(x)) if args.residual else pd.Series(0.0,index=y.index)
if args.schedule_fallback:
    assert args.residual and args.features=='nm'
    offset=nm_schedule_fallback_baseline(x)
target=y-offset
target=target.clip(-args.cap,args.cap) if args.cap else target
model=CatBoostRegressor(iterations=args.iterations,depth=args.depth,learning_rate=.055,loss_function='RMSE',random_seed=20260907,thread_count=6,allow_writing_files=False,verbose=200,task_type=args.device)
print('Start',args.name,'fit',int(fit.sum()),'validation',int(validation.sum()),'features',len(x.columns),flush=True)
model.fit(x.loc[fit],target.loc[fit],cat_features=category,eval_set=(x.loc[validation],(y-offset).loc[validation]),early_stopping_rounds=120)
pred=np.maximum(0,model.predict(x.loc[validation],thread_count=4)+offset.loc[validation])
result=data.loc[validation,[ID,'ADEP_mvt',TIME,TARGET]].copy()
result['prediction']=pred
model.save_model(str(out/'validation-model.cbm'))
result.to_parquet(out/'validation.parquet',index=False)
report={'status':'LOCAL_HOLDOUT_NOT_OFFICIAL_SCORE','experiment':vars(args),'rmse_seconds':rmse(y.loc[validation],pred),'selected_iterations':int(model.tree_count_),'training_rows':int(fit.sum()),'validation_rows':int(validation.sum()),'per_airport':{str(a):{'n':len(g),'rmse':rmse(g[TARGET],g.prediction)} for a,g in result.groupby('ADEP_mvt')},'per_month':{str(a):{'n':len(g),'rmse':rmse(g[TARGET],g.prediction)} for a,g in result.groupby(pd.to_datetime(result[TIME],utc=True).dt.month)},'features':list(x.columns),'categories':category,'feature_importance':dict(zip(x.columns,model.feature_importances_.tolist())),'input_hashes':{f.name:sha256(f) for f in paths},'elapsed_seconds':time.monotonic()-start}
write_json(out/'report.json',report)
print(json.dumps({k:report[k] for k in ['rmse_seconds','selected_iterations','per_airport','per_month','elapsed_seconds']}),flush=True)
if args.refit:
    final=CatBoostRegressor(iterations=model.tree_count_,depth=args.depth,learning_rate=.055,loss_function='RMSE',random_seed=20260907,thread_count=6,allow_writing_files=False,verbose=200,task_type=args.device)
    final.fit(x.loc[y.ge(0)],target.loc[y.ge(0)],cat_features=category)
    final.save_model(str(out/'model.cbm'))
    report['model_sha256']=sha256(out/'model.cbm')
    write_json(out/'report.json',report)
