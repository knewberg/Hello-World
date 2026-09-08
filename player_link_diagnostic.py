from pathlib import Path
import json, re
import pandas as pd

BASE='https://media.githubusercontent.com/media/JeffreyRStevens/ncaavolleyballr/refs/heads/main/data-csv'
OUT=Path('player_link_diagnostic_output'); OUT.mkdir(exist_ok=True)

def norm(x):
    x=str(x).lower().strip()
    x=x.replace('&','and')
    x=re.sub(r'\bsaint\b','st',x)
    x=re.sub(r'\bstate\b','st',x)
    x=re.sub(r'[^a-z0-9]+','',x)
    return x

report={}
all_unmatched=[]
for year in [2023,2024,2025]:
    t=pd.read_csv(f'{BASE}/wvb_teammatch_div1_{year}.csv',low_memory=False)
    p=pd.read_csv(f'{BASE}/wvb_playermatch_div1_{year}.csv',low_memory=False)
    tnames=sorted(t['Team'].dropna().astype(str).unique())
    pnames=sorted(p['Team'].dropna().astype(str).unique())
    tex=set(tnames); pex=set(pnames)
    tnorm={norm(x):x for x in tnames}; pnorm={norm(x):x for x in pnames}
    exact=sorted(tex&pex)
    normmatch=sorted(set(tnorm)&set(pnorm))
    unmatched_t=sorted(tex-pex)
    unmatched_after=[x for x in tnames if norm(x) not in pnorm]
    report[str(year)]={
      'team_columns':list(t.columns),'player_columns':list(p.columns),
      'team_rows':len(t),'player_rows':len(p),'team_unique':len(tnames),'player_unique':len(pnames),
      'exact_name_matches':len(exact),'normalized_name_matches':len(normmatch),
      'exact_coverage':len(exact)/len(tnames),'normalized_coverage':len(normmatch)/len(tnames),
      'team_names_sample':tnames[:30],'player_team_names_sample':pnames[:30],
      'player_head':p.head(5).where(pd.notna(p.head(5)),None).to_dict('records'),
      'unmatched_team_names_exact':unmatched_t,
      'unmatched_team_names_after_norm':unmatched_after,
      'normalized_alias_pairs':[{'team':tnorm[k],'player':pnorm[k]} for k in normmatch if tnorm[k]!=pnorm[k]],
    }
    for x in unmatched_after: all_unmatched.append({'year':year,'team':x,'norm':norm(x)})

(OUT/'diagnostic.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
pd.DataFrame(all_unmatched).to_csv(OUT/'unmatched_team_names.csv',index=False)
print(json.dumps({y:{k:v for k,v in d.items() if k not in ['player_head','unmatched_team_names_exact','unmatched_team_names_after_norm','normalized_alias_pairs','team_names_sample','player_team_names_sample','team_columns','player_columns']} for y,d in report.items()},indent=2))
