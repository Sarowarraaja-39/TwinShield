#!/usr/bin/env python3
"""Run this standard-Python notebook without Jupyter; never change its code cells."""
import ast
import base64
import contextlib
import ctypes
import ctypes.util
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time
import traceback
import warnings


def restrict_runtime():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    allowed_cpus = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, allowed_cpus)
    libname = ctypes.util.find_library('seccomp')
    if not libname:
        raise RuntimeError('libseccomp unavailable; refusing unfiltered execution')
    lib = ctypes.CDLL(libname, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    blocked = ['socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4',
               'sendto', 'sendmsg', 'sendmmsg', 'recvfrom', 'recvmsg', 'recvmmsg',
               'execve', 'execveat', 'fork', 'vfork', 'ptrace',
               'process_vm_readv', 'process_vm_writev', 'io_uring_setup']
    for name in blocked:
        syscall = lib.seccomp_syscall_resolve_name(name.encode())
        if syscall >= 0 and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, syscall, 0) != 0:
            raise RuntimeError('seccomp rule failed: ' + name)
    if lib.seccomp_load(ctx) != 0:
        raise RuntimeError('seccomp_load failed')
    lib.seccomp_release(ctx)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        network_denied = True
    else:
        sock.close()
        raise RuntimeError('network denial self-test failed')
    return {'network_socket_denied': network_denied, 'seccomp': True,
            'cpu_seconds_limit': 120, 'address_space_bytes_limit': 4 * 1024**3,
            'cpu_affinity': allowed_cpus,
            'filesystem_boundary': 'existing hosted environment; no additional mount namespace'}


def main():
    source = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    out = Path.cwd()
    raw = source.read_bytes()
    nb = json.loads(raw)
    original_code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    policy = restrict_runtime()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import seaborn
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    summary = {'mode': mode, 'input_sha256': hashlib.sha256(raw).hexdigest(),
               'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                           'pandas': pd.__version__, 'scipy': scipy.__version__,
                           'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
                           'seaborn': seaborn.__version__},
               'policy': policy, 'cells': [], 'figures': [], 'status': 'running'}
    current_outputs = []
    current_cell = 0
    def show(*args, **kwargs):
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
            payload = buffer.getvalue()
            filename = f'cell_{current_cell:02d}_figure_{len(summary["figures"])+1:02d}.png'
            (out / filename).write_bytes(payload)
            summary['figures'].append(filename)
            current_outputs.append({'output_type': 'display_data', 'metadata': {},
                                    'data': {'image/png': base64.b64encode(payload).decode(),
                                             'text/plain': [f'<Captured figure from original cell {current_cell}>']}})
            plt.close(fig)
    plt.show = show
    namespace = {'__name__': '__main__'}
    count = 0
    started = time.monotonic()
    for current_cell, cell in enumerate(nb['cells'], 1):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        cell['execution_count'] = count
        current_outputs = cell['outputs']
        stdout, stderr = io.StringIO(), io.StringIO()
        record = {'original_cell': current_cell, 'execution_count': count}
        t0 = time.monotonic()
        failure = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                tree = ast.parse(''.join(cell['source']), filename=f'original_cell_{current_cell}')
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, f'original_cell_{current_cell}', 'exec'), namespace)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), f'original_cell_{current_cell}', 'eval'), namespace)
                    if value is not None:
                        data = {'text/plain': [repr(value)]}
                        if isinstance(value, (pd.DataFrame, pd.Series)):
                            if isinstance(value, pd.DataFrame):
                                data['text/html'] = [value.to_html()]
                        current_outputs.append({'output_type': 'execute_result', 'execution_count': count,
                                                'metadata': {}, 'data': data})
            except Exception as exc:
                failure = {'ename': type(exc).__name__, 'evalue': str(exc),
                           'traceback': traceback.format_exc().splitlines()}
        if stdout.getvalue():
            current_outputs.insert(0, {'output_type': 'stream', 'name': 'stdout', 'text': [stdout.getvalue()]})
        warning_records = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
        if stderr.getvalue() or warning_records:
            text = stderr.getvalue() + ''.join(w['category']+': '+w['message']+'\n' for w in warning_records)
            current_outputs.append({'output_type': 'stream', 'name': 'stderr', 'text': [text]})
        record.update({'elapsed_seconds': round(time.monotonic()-t0, 4), 'warnings': warning_records,
                       'stdout': stdout.getvalue(), 'status': 'error' if failure else 'ok'})
        if failure:
            current_outputs.append({'output_type': 'error', **failure})
            record['error'] = failure
            summary['status'] = 'error'
        summary['cells'].append(record)
        print(f'CELL {current_cell:02d} {record["status"]} {record["elapsed_seconds"]:.3f}s', flush=True)
        if failure:
            break
    else:
        summary['status'] = 'ok'
    summary['elapsed_seconds'] = round(time.monotonic()-started, 4)
    summary['executed_code_cells'] = len(summary['cells'])
    summary['total_code_cells'] = len(original_code)
    if summary['status'] == 'ok':
        encoder = namespace['label_encoder']
        summary['observed_label_mapping'] = {str(k): int(v) for k,v in zip(encoder.classes_, encoder.transform(encoder.classes_))}
        summary['dataset_shape'] = list(namespace['df'].shape)
        summary['original_metrics_class1_is_normal'] = namespace['results_df'].reset_index().to_dict(orient='records')
        namespace['results_df'].to_csv(out/'synthetic_original_metrics_CLASS1_NORMAL.csv')
        summary['confusion_matrices_numeric_order_0_attack_1_normal'] = {}
        corrected = []
        for name, pred, proba in [('Baseline',namespace['y_pred_baseline'],namespace['y_proba_baseline']),
                                  ('TwinShield',namespace['y_pred_twinshield'],namespace['y_proba_twinshield'])]:
            truth = namespace['y_test']
            summary['confusion_matrices_numeric_order_0_attack_1_normal'][name] = confusion_matrix(truth,pred,labels=[0,1]).tolist()
            yt, yp = 1-truth, 1-pred
            tn,fp,fn,tp = confusion_matrix(yt,yp,labels=[0,1]).ravel()
            corrected.append({'Model':name, 'Accuracy':float(accuracy_score(yt,yp)),
                'Attack Precision':float(precision_score(yt,yp,zero_division=0)),
                'Attack Recall':float(recall_score(yt,yp,zero_division=0)),
                'Attack F1':float(f1_score(yt,yp,zero_division=0)),
                'False Alarm Rate On Normal':float(fp/(tn+fp)),
                'Attack ROC AUC':float(roc_auc_score(yt,1-proba))})
        summary['synthetic_metrics_reinterpreted_attack_positive'] = corrected
        pd.DataFrame(corrected).to_csv(out/'synthetic_attack_metrics_NOT_DISSERTATION_RESULTS.csv',index=False)
        probes=[]
        def probe(name, expected, observed, defect):
            probes.append({'name':name,'expected':expected,'observed':observed,'defect_reproduced':bool(defect)})
        mapping=summary['observed_label_mapping']
        probe('positive_label_contract',{'Normal':0,'Abnormal':1},mapping,mapping!={'Normal':0,'Abnormal':1})
        label_env={'df':pd.DataFrame({'Attack Type':['Normal Traffic','Normal Traffic ',None,'Unrecognised Label']})}
        binary_tree=ast.parse(''.join(nb['cells'][16]['source'])).body[0]
        exec(compile(ast.Module(body=[binary_tree],type_ignores=[]),'original_cell_17_assignment','exec'),label_env)
        observed=label_env['df']['Binary Label'].tolist()
        probe('unknown_and_missing_label_handling','Reject missing/unknown labels; normalise accepted labels',observed,observed==['Normal','Abnormal','Abnormal','Abnormal'])
        raw=namespace['X_train'].drop(columns=['Flow Duration']).head(3)
        representation=namespace['twin'].represent(raw)
        probe('missing_tracked_feature','Clear missing-feature error',list(representation.columns),len(representation.columns)==13)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                namespace['evaluate_model']([0,0],[0,0],'single_class')
        except Exception as exc:
            single=f'{type(exc).__name__}: {exc}'
            defect=True
        else:
            single='returned without exception';defect=False
        probe('single_class_evaluation','Explicit [0,1] matrix and defined unavailable-rate policy',single,defect)
        dirty=namespace['X_train'].head(20).copy()
        dirty.iloc[0,0]=np.inf
        try:
            from sklearn.preprocessing import StandardScaler
            StandardScaler().fit_transform(dirty)
        except Exception as exc:
            inf=f'{type(exc).__name__}: {exc}';defect=True
        else:
            inf='returned without exception';defect=False
        probe('infinite_numeric_feature','Reject or repair invalid numeric input before model fitting',inf,defect)
        summary['boundary_probes']=probes
        (out/'boundary_probes.json').write_text(json.dumps(probes,indent=2))
    assert original_code == [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    summary['code_cells_unchanged']=True
    banner=('SYNTHETIC TEST DATA ONLY. This is not a CICIDS2017 experiment and its scores must not be used in the dissertation.'
            if mode=='synthetic' else 'Original code, no dataset supplied. Execution stops at the first runtime error; old outputs were cleared.')
    nb['cells'].insert(0,{'cell_type':'markdown','id':'audit-provenance','metadata':{},'source':['# Execution audit — '+mode+'\n\n'+banner+'\n\nOriginal code-cell sources are unchanged. Cell numbers in the report refer to the supplied notebook.']})
    nb['metadata']['execution_audit']={k:summary[k] for k in ['mode','input_sha256','runtime','policy','status','code_cells_unchanged']}
    (out/f'original_code_{mode.upper()}_EXECUTED.ipynb').write_text(json.dumps(nb,indent=1))
    (out/'execution_summary.json').write_text(json.dumps(summary,indent=2))
    log=[]
    for c in summary['cells']:
        log.append(f'CELL {c["original_cell"]}: {c["status"]}\n{c["stdout"]}')
        log.extend(f'{w["category"]}: {w["message"]}' for w in c['warnings'])
        if c.get('error'): log.extend(c['error']['traceback'])
    (out/'cell_outputs.txt').write_text('\n'.join(log))
    print('FINAL',summary['status'],'cells',summary['executed_code_cells'],'/',summary['total_code_cells'],flush=True)
    return 0 if summary['status']=='ok' else 1

if __name__=='__main__':
    raise SystemExit(main())
