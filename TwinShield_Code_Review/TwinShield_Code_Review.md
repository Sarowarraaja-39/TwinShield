# TwinShield notebook code and execution review

Reviewed on 26 October 2026. Source: `Dissertation.ipynb`, 48 cells, including 29 code cells.

**Verdict:** execution against the real dataset is **inconclusive** because the CSV was not supplied. Code quality **fails the stated attack-positive evaluation contract** because the encoded class meanings are reversed. All 29 unchanged code cells executed successfully on a clearly labelled artificial fixture; this establishes code-path coverage on that fixture, not research validity.

## What actually ran

| Check | Fresh observed result | Evidence |
| --- | --- | --- |
| Parse every code cell | 29/29 accepted by Python AST parser | `source/original_source.py`; same cell sources as original |
| Original notebook, no input fixture | Exit 1; original cell 4 fails with FileNotFoundError | `original/process_result.json`, `original/execution_summary.json` |
| Original notebook, synthetic fixture | Exit 0; 29/29 cells completed; 9 figures captured | `synthetic/process_result.json`, `synthetic/execution_summary.json`, executed notebook |
| Original label mapping | Abnormal = 0; Normal = 1 | Synthetic execution of original cell 25 |
| Boundary probes | Five edge cases exercised; observations below | `synthetic/boundary_probes.json` |
| Real CICIDS2017 model performance | Not reproduced | Exact cleaned CSV unavailable |
| Source preservation | Original file and code cells unchanged | SHA-256 and source comparisons in `verification.json` |

The original exception is:

```text
FileNotFoundError: [Errno 2] No such file or directory:
C:\Users\admin\Downloads\cicids2017_cleaned.csv
```

The notebook contains previously saved results for 2,520,751 rows and 53 columns. Those results are historical evidence, not outputs of this fresh execution. The dataset's saved schema includes 52 numeric predictors and seven traffic categories. I did not substitute a public dataset with different cleaning, labels, or feature selection.

## Findings

All locations below refer to the supplied notebook's original cell numbers; line references refer to the included `source/original_source.py`. The two executed copies add a provenance notice before the original cells.

### 1. Attack and normal class meanings are reversed — high severity, high confidence

**Axis/category:** Quality / correctness and reliability. **Location:** cell 25, line 171; evaluation cell 37, line 283; figures/report cells 39 and 41, lines 308 and 320; probability selection cells 35 and 43, lines 270 and 325.

The comment promises Normal = 0 and Abnormal = 1. `LabelEncoder` actually produces `{'Abnormal': 0, 'Normal': 1}`, confirmed by both the original saved output and fresh execution. Default binary precision, recall and F1 therefore measure **normal traffic**, and column 1 of `predict_proba` is the normal probability. The confusion-matrix axis labels and classification-report names are swapped.

The reported false-positive rate counts attacks predicted normal divided by all attacks: operationally this is the **missed-attack rate**, not the false-alarm rate on benign traffic. Accuracy is invariant to this consistent relabelling; it does not need to be inverted.

**Follow-up:** explicitly map Normal to 0 and Abnormal to 1, make metric labels explicit, select probability columns by `model.classes_`, and rerun every affected output. Do not only rename the plot axes while leaving the metric semantics unchanged. Scikit-learn's version-matched documentation confirms the positive-label default and matrix ordering [1–3].

### 2. The full experiment cannot be reproduced from the delivered notebook — high severity, high confidence

**Axis/category:** Contract / coverage and compatibility. **Location:** cell 2, line 26; cell 4, line 30.

The CSV points to one Windows user's Downloads folder. A fresh run fails at `pd.read_csv`. No matching CSV was supplied or located in the available file search. The cleaned data and cleaning procedure are necessary to reproduce this particular 53-column derivative; the official raw dataset is not an interchangeable input.

**Follow-up:** supply the exact cleaned CSV plus provenance, cleaning steps, checksum, and a configurable path. Save package versions and the train/test assignment with the final experiment. The original source and its historical results remain unchanged in this review.

### 3. The experiment cannot isolate a Digital Twin benefit — high severity, high confidence

**Axis/category:** Quality / research scope and coverage. **Location:** cells 30, 33 and 35; lines 229, 252 and 260.

The baseline uses Logistic Regression on 52 features without class weighting. TwinShield uses a 200-tree Random Forest on 14 features with balanced class weights. Three experimental factors change together. A difference in their scores cannot establish that the feature-representation layer caused the improvement.

The implemented `DigitalTwin.represent` is a static column selector. No per-asset state, time synchronisation, state evolution, physical/network simulation, counterfactual model, or live feedback loop appears in the reviewed code. It is a reduced-feature IDS prototype; a stronger digital-twin claim needs an explicit definition and supporting implementation/evidence.

**Follow-up:** evaluate the same classifiers on full and reduced features under the same weighting and split protocol, reporting each factor separately. Treat those as planned experiments; no comparative performance claim is established by the synthetic run.

### 4. Random row splitting leaves leakage and generalisation questions unresolved — medium severity, medium confidence

**Axis/category:** Quality / coverage. **Location:** cell 10, line 67; cell 27, line 190.

The saved notebook reports 161 exact duplicate rows but does not remove or group them. A stratified random row split does not prevent duplicate feature vectors or related flows from crossing partitions. Whether that happened in this dataset is **unverified**, not demonstrated here. The source CICIDS2017 data span several capture days and scenarios [4], making capture/session-aware evaluation relevant.

**Follow-up:** check feature-vector and group overlap between partitions, resolve conflicting duplicate labels explicitly, and retain capture/session/time metadata when building the cleaned dataset. Use an appropriate held-out capture or session where possible, report its class coverage, and keep the random split as a separately described in-distribution result. The saved 53-column schema does not contain enough provenance to reconstruct every grouping decision.

### 5. Missing tracked features are silently dropped — medium severity, high confidence

**Axis/category:** Quality / reliability. **Location:** cell 30, line 231; cell 48, line 378.

Dropping `Flow Duration` from an input returns 13 features without an error, while `twin.feature_columns` still lists 14. This makes the interface schema inconsistent, can fail a fitted scaler/model, and can cause a length mismatch when indexing feature importances.

**Follow-up:** validate the required columns at the input boundary, preserve a fixed order, and report which are missing. Use the actual fitted feature names when producing importances.

### 6. Unknown or missing traffic labels are treated as attacks — medium severity, high confidence

**Axis/category:** Quality / reliability. **Location:** cell 17, line 96.

Fresh boundary inputs `Normal Traffic `, a missing value, and `Unrecognised Label` all map to Abnormal. This conflates dirty or unsupported labels with confirmed attacks. The supplied saved dataset reports no missing values; this finding concerns how new or changed input is handled.

**Follow-up:** validate the accepted label vocabulary, trim approved formatting differences, and reject missing/unknown labels with an informative error.

### 7. Evaluation and preprocessing lack explicit policies for boundary inputs — medium severity, high confidence

**Axis/category:** Quality / reliability. **Location:** cells 8–9 and 28, lines 49–62 and 201; cell 37, line 277.

`evaluate_model([0, 0], [0, 0], ...)` raises `ValueError: not enough values to unpack (expected 4, got 1)`. With a missing negative class, false-positive-rate division also needs an explicit undefined-value policy. An infinite numeric feature is diagnosed but not repaired or rejected early; the scaling stage then raises a ValueError. These boundary conditions were not present in the successful synthetic fixture or claimed to occur in the supplied saved dataset.

**Follow-up:** request a fixed `[0, 1]` confusion matrix and define unavailable metrics explicitly. Validate non-finite predictors before fitting; if imputing missing values, fit that preprocessing on training data only. Do not silently impute target labels.

### 8. Plotting warnings and resource/reproducibility limits — low severity, high confidence

**Axis/category:** Quality / compatibility and coverage. **Location:** cells 14 and 19; cells 23, 28, 31 and 35.

The fresh run emits seaborn palette-without-hue warnings and boxplot API deprecation warnings. These did not prevent execution. The full saved dataset occupies about 1 GB before training; dataframe copies, full and reduced scaled arrays, density plots, and 200 unrestricted trees require additional resources. Full-dataset runtime and peak memory were not measured. A seed helps reproducibility but does not replace version and dataset records.

**Follow-up:** update plotting calls for the supported library versions, sample data only for expensive visualisation when justified, and measure training time and peak memory on the real dataset. Keep the full evaluation population separate from plotting samples.

## Correct interpretation of the original saved counts

**These are recalculations from the original saved confusion-matrix image, not newly trained results.** The image is included as `saved_confusion_matrices_ORIGINAL_MISLABELLED.png`; its numeric counts were visually inspected. Its first row/column represents attacks and its second represents normal traffic, despite the displayed names.

| Model | Accuracy | Attack precision | Attack recall | Attack F1 | False alarms / normal |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 95.5328% | 88.0101% | 85.1471% | 86.5550% | 2.3570% |
| TwinShield Random Forest | 99.7658% | 99.2196% | 99.3949% | 99.3072% | 0.1589% |

For TwinShield, the saved counts mean **105,780 attacks detected, 644 attacks missed, 832 normal flows incorrectly alerted, and 522,932 normal flows correctly accepted**. The test set contains 106,424 attack flows and 523,764 normal flows; this also exposes the swapped support labels in the original classification report.

With attacks treated as positive:

| TwinShield true class | Predicted normal | Predicted attack |
| --- | --- | --- |
| Normal | 522,932 | 832 |
| Attack | 644 | 105,780 |

```text
Attack precision = 105780 / (105780 + 832)
Attack recall    = 105780 / (105780 + 644)
False alarm rate = 832 / (522932 + 832)
```

ROC AUC is not automatically `1 - old_AUC` when both the positive label and its probability are reversed consistently. The numeric AUC can remain the same. The alert-class ROC curve should still be regenerated from the actual predictions. Raw saved predictions/probabilities were not supplied, so that regeneration was not possible here.

## Fresh synthetic run: code execution only

The fixture has 2,400 generated rows plus 16 deliberate duplicate rows and the same 52 feature names. It includes the seven saved label names. Values come from `make_classification` (seed 20261026) and **do not represent physically validated network flows**. The original notebook's code, model hyperparameters and seed remain unchanged; the test fixture is staged at the literal filename requested by the code inside its working directory on Linux.

All 29 code cells completed in 2.55 seconds after library initialisation. Nine plots were captured. The notebook retains its original label bug, and the execution records show it. To avoid misrepresenting that bug, the separate CSV below reinterprets its predictions with attacks as positive:

| Synthetic model | Accuracy | Attack precision | Attack recall |
| --- | --- | --- | --- |
| Baseline | 87.75% | 72.37% | 50.93% |
| TwinShield | 83.28% | 65.22% | 13.89% |

These artificial-fixture values are included for traceability only. They must not be reported as CICIDS2017 performance or used to rank the models. The poor attack recall on the artificial fixture does not prove poor real-dataset performance.

Files: `synthetic/original_code_SYNTHETIC_EXECUTED.ipynb`, `synthetic/cell_outputs.txt`, `synthetic/synthetic_attack_metrics_NOT_DISSERTATION_RESULTS.csv`, and `synthetic/boundary_probes.json`. The synthetic notebook's original plots intentionally preserve its existing class-labelling defects; its provenance notice and filename distinguish it from a corrected experiment.

## Execution method and limits

The environment had no Jupyter/IPython kernel installed. A standard-Python cell runner executed code in source order and captured final expressions, stdout, warnings, errors and matplotlib figures. The reviewed cells contain no notebook magics. This verifies their Python code paths; it does not verify Jupyter frontend rendering or widget behaviour.

Each run used a fresh child process, a minimal environment, its own working directory, two available CPU cores, a 120 CPU-second limit, a 150-second wall timeout and a 4 GiB address-space limit. Kernel seccomp rules denied network sockets and external program execution; the socket-denial probe returned PermissionError in both runs. The existing hosted environment supplies the filesystem boundary: no additional mount namespace was available. This is not a claim of a separately provisioned VM or unrestricted malware sandbox.

Runtime: Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0; matplotlib 3.10.8; seaborn 0.13.2. No packages were installed for this audit.

No direct shell commands, credential-reading code, remote requests or unsafe model-deserialisation calls were found in the reviewed notebook cells. That source inspection is not an audit of all installed dependencies.

## Recommended next step

Supply the exact `cicids2017_cleaned.csv` used for the saved experiment. Then correct the explicit label contract, refresh all outputs, validate the split and run controlled classifier/feature ablations before making an improvement claim. The current audit does not modify the source notebook or implement these recommendations.

After validating the source traffic labels, the central correction to cell 25 is an explicit mapping:

```python
df["Label_Encoded"] = df["Binary Label"].map(
    {"Normal": 0, "Abnormal": 1}
).astype("int8")
```

This is a recommended change, not a claim that a corrected real-data experiment has already been run.

Audit source SHA-256:

```text
a802b679c044b5b63dcefb9fbef2adf16d9ef43f3a46a9a9ad7ab36db7cfcb08
```

## Primary references

1. [scikit-learn 1.8: precision_score — positive label defaults to 1](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.precision_score.html).
2. [scikit-learn 1.8: confusion_matrix — label order](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.confusion_matrix.html).
3. [scikit-learn 1.8: roc_curve — positive-label scores](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_curve.html).
4. [University of New Brunswick: CIC-IDS2017 — dataset provenance and capture days](https://www.unb.ca/cic/datasets/ids-2017.html).
