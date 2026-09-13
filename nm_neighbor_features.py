# SPDX-License-Identifier: GPL-3.0-only
"""Earlier NM departure proxies; departure targets and block times are excluded."""
from __future__ import annotations
import numpy as np,pandas as pd
def norm(s: pd.Series) -> pd.Series:return s.astype('string').str.strip().str.upper().replace('',pd.NA).fillna('UNKNOWN').astype(str)
def completed_nm_neighbors(dep: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    cols=['ADEP_mvt','ADEP_flt','RUNWAY_mvt','STAND_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt']
    pool=context.loc[context.PHASE_mvt.eq('DEP'),cols].copy()
    mv=pd.to_datetime(pool.MVT_TIME_UTC_mvt,utc=True,errors='coerce')
    nm=pd.to_datetime(pool.AOBT_3_flt,utc=True,errors='coerce');proxy=(mv-nm).dt.total_seconds()
    valid=mv.notna()&nm.notna()&proxy.between(0,7200)&norm(pool.ADEP_mvt).eq(norm(pool.ADEP_flt))
    pool=pool.loc[valid].copy();mv=mv.loc[valid];pool['proxy']=proxy.loc[valid]
    pool['seconds']=mv.dt.as_unit('ns').astype('int64')/1e9
    pool['period']=mv.dt.strftime('%Y-%m');pool['airport']=norm(pool.ADEP_mvt)
    pool['runway']=norm(pool.RUNWAY_mvt);pool['stand']=norm(pool.STAND_mvt)
    clock=pd.to_datetime(dep.MVT_TIME_UTC_mvt,utc=True,errors='raise')
    if clock.isna().any():raise ValueError('Query times must be complete')
    query=clock.dt.as_unit('ns').astype('int64').to_numpy()/1e9
    own=(clock-pd.to_datetime(dep.AOBT_3_flt,utc=True,errors='coerce')).dt.total_seconds()
    own_values=own.where(own.between(0,7200)&norm(dep.ADEP_mvt).eq(norm(dep.ADEP_flt))).to_numpy()
    keys=pd.DataFrame({'airport':norm(dep.ADEP_mvt),'runway':norm(dep.RUNWAY_mvt),
        'stand':norm(dep.STAND_mvt),'period':clock.dt.strftime('%Y-%m')},index=dep.index)
    out=pd.DataFrame(index=dep.index)
    for label,group in [('airport',['airport','period']),('runway',['airport','period','runway']),('stand',['airport','period','stand'])]:
        scoped=pool if label=='airport' else pool.loc[pool[label].ne('UNKNOWN')]
        lookup={key:g.sort_values(['seconds','proxy'],kind='stable') for key,g in scoped.groupby(group,dropna=False)}
        result={f'{window}m_{metric}':np.full(len(dep),np.nan) for window in [15,60] for metric in ['mean','std','own_minus_mean']}
        for window in [15,60]:result[f'{window}m_count']=np.zeros(len(dep))
        result['last_age_seconds']=np.full(len(dep),np.nan);result['last_proxy']=np.full(len(dep),np.nan)
        for key,indices in keys.groupby(group,dropna=False).indices.items():
            g=lookup.get(key if isinstance(key,tuple) else (key,))
            if g is None or g.empty:continue
            t=g.seconds.to_numpy();v=g.proxy.to_numpy();q=query[indices];right=np.searchsorted(t,q,side='left')
            sums=np.r_[0.,np.cumsum(v)];sq=np.r_[0.,np.cumsum(v*v)]
            for window in [15,60]:
                left=np.searchsorted(t,q-window*60,side='left');n=right-left
                mean=np.divide(sums[right]-sums[left],n,out=np.full(len(n),np.nan),where=n>0)
                second=np.divide(sq[right]-sq[left],n,out=np.full(len(n),np.nan),where=n>0)
                result[f'{window}m_count'][indices]=n;result[f'{window}m_mean'][indices]=mean
                result[f'{window}m_std'][indices]=np.sqrt(np.maximum(0,second-mean*mean))
                result[f'{window}m_own_minus_mean'][indices]=own_values[indices]-mean
            last=np.maximum(0,right-1);recent=(right>0)&((q-t[last])<=7200)
            # Aggregate all events tied at the most recent timestamp, invariant to row order.
            tie_start=np.searchsorted(t,t[last],side='left');tie_end=np.searchsorted(t,t[last],side='right')
            last_mean=(sums[tie_end]-sums[tie_start])/(tie_end-tie_start)
            result['last_age_seconds'][indices[recent]]=q[recent]-t[last[recent]]
            result['last_proxy'][indices[recent]]=last_mean[recent]
        for metric,v in result.items():out[f'nm_neighbor_{label}_{metric}']=v.astype('float32')
    return out
