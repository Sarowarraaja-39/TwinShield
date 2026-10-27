#!/usr/bin/env python3
"""Create a labelled synthetic fixture, then run independent bounded child processes."""
import json
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

def main():
    if sys.platform != 'linux':
        raise RuntimeError('This audit harness requires Linux and libseccomp; it is not a cross-platform installer.')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,default=Path(__file__).resolve().parent.parent/'source/Dissertation_original.ipynb')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'rerun')
    args=parser.parse_args()
    SOURCE=args.notebook.resolve()
    DELIVERY=args.output.resolve()
    RUNNER=Path(__file__).with_name('run_notebook_audit.py')
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    nb=json.loads(SOURCE.read_text())
    info=''.join(nb['cells'][5]['outputs'][0]['text'])
    columns=[]
    for line in info.splitlines():
        m=re.match(r'^\s*\d+\s+(.+?)\s+(int64|float64|object)\s*$',line)
        if m and m.group(2)!='object': columns.append(m.group(1).strip())
    if len(columns)!=52:
        raise RuntimeError(f'Expected 52 numeric columns from saved schema, got {len(columns)}')
    matrix,target=make_classification(n_samples=2400,n_features=52,n_informative=12,
        n_redundant=12,n_clusters_per_class=2,weights=[0.83,0.17],
        class_sep=1.0,flip_y=0.02,random_state=20261026)
    matrix=np.abs(matrix*10+50)
    fixture=pd.DataFrame(matrix,columns=columns)
    attacks=np.array(['DoS','DDoS','Port Scanning','Brute Force','Web Attacks','Bots'])
    labels=np.full(len(target),'Normal Traffic',dtype=object)
    labels[target==1]=attacks[np.arange((target==1).sum())%len(attacks)]
    fixture['Attack Type']=labels
    fixture=pd.concat([fixture,fixture.head(16)],ignore_index=True)
    run_results=[]
    for mode in ['original','synthetic']:
        work=DELIVERY/mode
        work.mkdir(parents=True,exist_ok=True)
        for d in ['home','tmp','mplconfig']:
            (work/d).mkdir(exist_ok=True)
        if mode=='synthetic':
            fixture.to_csv(work/r'C:\Users\admin\Downloads\cicids2017_cleaned.csv',index=False)
            (work/'SYNTHETIC_DATA_NOTICE.txt').write_text(
                'Artificial numeric fixture for code-path testing only. NOT CICIDS2017.\n'
                'Generated with sklearn.datasets.make_classification; seed=20261026; 2400 rows plus 16 deliberate duplicates.\n'
                'Feature names match the saved schema; values do not model physically valid network flows.\n')
        env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(work/'home'),
             'TMPDIR':str(work/'tmp'),'MPLCONFIGDIR':str(work/'mplconfig'),
             'MPLBACKEND':'Agg','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
             'MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONHASHSEED':'42'}
        command=[sys.executable,str(RUNNER),str(SOURCE),mode]
        try:
            result=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=150)
            entry={'mode':mode,'command':command,'cwd':str(work),'exit_code':result.returncode,
                   'stdout':result.stdout,'stderr':result.stderr,'wall_timeout_seconds':150}
        except subprocess.TimeoutExpired as exc:
            entry={'mode':mode,'command':command,'cwd':str(work),'exit_code':None,
                   'status':'timeout','stdout':str(exc.stdout),'stderr':str(exc.stderr),
                   'wall_timeout_seconds':150}
        (work/'process_result.json').write_text(json.dumps(entry,indent=2))
        run_results.append(entry)
        print(json.dumps(entry),flush=True)
    (DELIVERY/'run_results.json').write_text(json.dumps(run_results,indent=2))
    scripts=DELIVERY/'audit_scripts'
    scripts.mkdir(exist_ok=True)
    shutil.copyfile(RUNNER,scripts/RUNNER.name)
    shutil.copyfile(__file__,scripts/Path(__file__).name)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
