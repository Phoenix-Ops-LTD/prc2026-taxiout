# SPDX-License-Identifier: GPL-3.0-only
"""Following NM departure proxies for retrospective supplied-clock context.

Departure labels and block times are excluded. These observations do not
establish availability before departure. Windows stay within each UTC month.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def norm(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.upper().replace('',pd.NA).fillna('UNKNOWN').astype(str)

def following_nm_neighbors(dep: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    columns=['ADEP_mvt','ADEP_flt','RUNWAY_mvt','STAND_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt']
    pool=context.loc[context.PHASE_mvt.eq('DEP'),columns].copy()
    moved=pd.to_datetime(pool.MVT_TIME_UTC_mvt,utc=True,errors='coerce')
    actual=pd.to_datetime(pool.AOBT_3_flt,utc=True,errors='coerce')
    proxy=(moved-actual).dt.total_seconds()
    valid=moved.notna() & actual.notna() & proxy.between(0,7200) & norm(pool.ADEP_mvt).eq(norm(pool.ADEP_flt))
    pool=pool.loc[valid].copy(); moved=moved.loc[valid]
    pool['proxy']=proxy.loc[valid].astype(float)
    pool['seconds']=moved.dt.as_unit('ns').astype('int64')/1e9
    pool['period']=moved.dt.strftime('%Y-%m')
    for key, column in [('airport','ADEP_mvt'),('runway','RUNWAY_mvt'),('stand','STAND_mvt')]: pool[key]=norm(pool[column])
    clock=pd.to_datetime(dep.MVT_TIME_UTC_mvt,utc=True,errors='raise')
    if clock.isna().any(): raise ValueError('Query timestamps must be complete')
    query=clock.dt.as_unit('ns').astype('int64').to_numpy()/1e9
    own=(clock-pd.to_datetime(dep.AOBT_3_flt,utc=True,errors='coerce')).dt.total_seconds()
    own_values=own.where(own.between(0,7200) & norm(dep.ADEP_mvt).eq(norm(dep.ADEP_flt))).to_numpy()
    keys=pd.DataFrame({'airport':norm(dep.ADEP_mvt),'runway':norm(dep.RUNWAY_mvt),
        'stand':norm(dep.STAND_mvt),'period':clock.dt.strftime('%Y-%m')},index=dep.index)
    output=pd.DataFrame(index=dep.index)
    for label, group in [('airport',['airport','period']),('runway',['airport','period','runway']),('stand',['airport','period','stand'])]:
        scoped=pool if label=='airport' else pool.loc[pool[label].ne('UNKNOWN')]
        lookup={key:rows.sort_values(['seconds','proxy'],kind='stable') for key,rows in scoped.groupby(group,dropna=False)}
        result={f'{window}m_{metric}':np.full(len(dep),np.nan) for window in [15,60] for metric in ['mean','std','own_minus_mean']}
        for window in [15,60]: result[f'{window}m_count']=np.zeros(len(dep))
        result['next_wait_seconds']=np.full(len(dep),np.nan); result['next_proxy']=np.full(len(dep),np.nan)
        for group_key, indices in keys.groupby(group,dropna=False).indices.items():
            rows=lookup.get(group_key if isinstance(group_key,tuple) else (group_key,))
            if rows is None or rows.empty: continue
            t=rows.seconds.to_numpy(); values=rows.proxy.to_numpy(); q=query[indices]
            # Exclude the query itself and every movement at the exact query time.
            left=np.searchsorted(t,q,side='right')
            sums=np.r_[0.,np.cumsum(values)]; squares=np.r_[0.,np.cumsum(values*values)]
            for window in [15,60]:
                right=np.searchsorted(t,q+window*60,side='right'); count=right-left
                mean=np.divide(sums[right]-sums[left],count,out=np.full(len(count),np.nan),where=count>0)
                second=np.divide(squares[right]-squares[left],count,out=np.full(len(count),np.nan),where=count>0)
                result[f'{window}m_count'][indices]=count; result[f'{window}m_mean'][indices]=mean
                result[f'{window}m_std'][indices]=np.sqrt(np.maximum(0,second-mean*mean))
                result[f'{window}m_own_minus_mean'][indices]=own_values[indices]-mean
            nxt=np.minimum(left,len(t)-1); available=(left<len(t)) & ((t[nxt]-q)<=7200)
            start=np.searchsorted(t,t[nxt],side='left'); end=np.searchsorted(t,t[nxt],side='right')
            tied_mean=(sums[end]-sums[start])/(end-start)
            result['next_wait_seconds'][indices[available]]=t[nxt[available]]-q[available]
            result['next_proxy'][indices[available]]=tied_mean[available]
        for metric, values in result.items(): output[f'nm_following_{label}_{metric}']=values.astype('float32')
    return output
