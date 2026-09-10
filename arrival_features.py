# SPDX-License-Identifier: GPL-3.0-only
"""ARR-only completed ground observations explicitly supplied by PRC.

Official schema removes taxi/block values only for DEP rows:
https://prc-data-challenge-2026.netlify.app/data.html#the-ranking-dataset
Filter arrivals before reading taxi/block fields; departure targets never enter features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def normalized(series:pd.Series)->pd.Series:
    return series.astype('string').str.strip().str.upper().replace('',pd.NA).fillna('UNKNOWN').astype(str)

def completed_arrival_features(dep:pd.DataFrame,context:pd.DataFrame)->pd.DataFrame:
    # Filter FIRST: identically named target/block fields are supplied inputs on
    # ARR rows, and forbidden prediction targets on DEP rows.
    cols=['ADES_mvt','RUNWAY_mvt','STAND_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt',
          'TAXITIME_SEC_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt']
    arr=context.loc[context.PHASE_mvt.eq('ARR'),cols].copy()
    landed=pd.to_datetime(arr.MVT_TIME_UTC_mvt,utc=True,errors='coerce')
    blocked=pd.to_datetime(arr.BLOCK_TIME_UTC_mvt,utc=True,errors='coerce')
    taxi=pd.to_numeric(arr.TAXITIME_SEC_mvt,errors='coerce')
    valid=landed.notna()&blocked.notna()&blocked.ge(landed)&taxi.between(0,7200)
    arr=arr.loc[valid].copy();blocked=blocked.loc[valid];taxi=taxi.loc[valid]
    arr['completed_seconds']=blocked.dt.as_unit('ns').astype('int64')/1e9
    arr['taxi_seconds']=taxi.astype(float)
    arr['airport']=normalized(arr.ADES_mvt);arr['runway']=normalized(arr.RUNWAY_mvt);arr['stand']=normalized(arr.STAND_mvt)
    arr['aircraft']=normalized(arr.AIRCRAFT_TYPE_mvt)
    arr['carrier']=normalized(arr.FLIGHT_mvt).str.extract(r'^([A-Z]{2,4})',expand=False).fillna('UNKNOWN')
    clock=pd.to_datetime(dep.MVT_TIME_UTC_mvt,utc=True,errors='raise')
    if clock.isna().any():raise ValueError('Departure movement timestamps must be complete')
    query=clock.dt.as_unit('ns').astype('int64').to_numpy()/1e9
    keys=pd.DataFrame({'airport':normalized(dep.ADEP_mvt),'runway':normalized(dep.RUNWAY_mvt),'stand':normalized(dep.STAND_mvt)},index=dep.index)
    out=pd.DataFrame(index=dep.index)
    for label,group_columns in [('airport',['airport']),('runway',['airport','runway'])]:
        output={f'{minutes}m_{metric}':np.full(len(dep),np.nan) for minutes in [15,30,60] for metric in ['mean','std']}
        output.update({f'{minutes}m_count':np.zeros(len(dep)) for minutes in [15,30,60]})
        output['last_age_seconds']=np.full(len(dep),np.nan);output['last_taxi_seconds']=np.full(len(dep),np.nan)
        lookup={key if isinstance(key,tuple) else (key,):g.sort_values('completed_seconds',kind='stable') for key,g in arr.groupby(group_columns,dropna=False)}
        for key,indices in keys.groupby(group_columns,dropna=False).indices.items():
            g=lookup.get(key if isinstance(key,tuple) else (key,))
            if g is None or g.empty:continue
            t=g.completed_seconds.to_numpy();v=g.taxi_seconds.to_numpy();q=query[indices]
            right=np.searchsorted(t,q,side='left')
            sums=np.r_[0.,np.cumsum(v)];squares=np.r_[0.,np.cumsum(v*v)]
            for minutes in [15,30,60]:
                left=np.searchsorted(t,q-minutes*60,side='left');n=right-left
                mean=np.divide(sums[right]-sums[left],n,out=np.full(len(n),np.nan),where=n>0)
                second=np.divide(squares[right]-squares[left],n,out=np.full(len(n),np.nan),where=n>0)
                output[f'{minutes}m_count'][indices]=n
                output[f'{minutes}m_mean'][indices]=mean
                output[f'{minutes}m_std'][indices]=np.sqrt(np.maximum(0,second-mean*mean))
            # Simultaneous completions have no defensible record order. Use
            # their mean for the last-completion measurement.
            last=g.groupby('completed_seconds').taxi_seconds.mean()
            last_times=last.index.to_numpy();last_right=np.searchsorted(last_times,q,side='left')
            previous=np.maximum(last_right-1,0);age=q-last_times[previous];recent=(last_right>0)&(age<=7200)
            output['last_age_seconds'][indices[recent]]=age[recent]
            output['last_taxi_seconds'][indices[recent]]=last.to_numpy()[previous[recent]]
        for name,values in output.items():out[f'completed_arrival_{label}_{name}']=values.astype('float32')
    stand_output={name:np.full(len(dep),np.nan) for name in ['last_age_seconds','last_taxi_seconds','aircraft_equal','carrier_equal']}
    lookup={key:g.sort_values('completed_seconds',kind='stable') for key,g in arr.groupby(['airport','stand'],dropna=False) if key[1]!='UNKNOWN'}
    aircraft=normalized(dep.AIRCRAFT_TYPE_mvt).to_numpy()
    carrier=normalized(dep.FLIGHT_mvt).str.extract(r'^([A-Z]{2,4})',expand=False).fillna('UNKNOWN').to_numpy()
    for key,indices in keys.groupby(['airport','stand'],dropna=False).indices.items():
        g=lookup.get(key if isinstance(key,tuple) else (key,))
        if g is None or g.empty:continue
        t=g.completed_seconds.to_numpy();right=np.searchsorted(t,query[indices],side='left');previous=np.maximum(right-1,0)
        ambiguous=g.completed_seconds.duplicated(keep=False).to_numpy()
        age=query[indices]-t[previous];recent=(right>0)&(age<=86400)&~ambiguous[previous]
        dst=indices[recent];src=previous[recent]
        stand_output['last_age_seconds'][dst]=age[recent]
        stand_output['last_taxi_seconds'][dst]=g.taxi_seconds.to_numpy()[src]
        stand_output['aircraft_equal'][dst]=(aircraft[dst]==g.aircraft.to_numpy()[src]).astype(float)
        stand_output['carrier_equal'][dst]=(carrier[dst]==g.carrier.to_numpy()[src]).astype(float)
    for name,values in stand_output.items():out['completed_arrival_stand_'+name]=values.astype('float32')
    return out
