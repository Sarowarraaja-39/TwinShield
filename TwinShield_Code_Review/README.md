# TwinShield execution audit

Start with TwinShield_Code_Review.md. This is an audit package, not a fixed application.

- source/: unchanged original notebook and a plain Python export with original cell markers.
- original/: fresh failed run with no CSV; old outputs were cleared.
- synthetic/: full code-path run on artificial data, plots, raw logs and boundary probes.
- saved_output_reinterpretation*: corrected attack-class interpretation of saved confusion counts; not a fresh benchmark.
- audit_scripts/: Linux-only audit harness. No package installer is included.

To repeat the synthetic code-path audit in a compatible Linux environment with libseccomp and the versions in the report:

```bash
python3 audit_scripts/launch_audit.py --output ./rerun
```

This deliberately reruns the original code and reproduces its defects. The launcher creates the synthetic fixture only in its own run directory. The supplied cleanly named synthetic CSV is not the real CICIDS2017 dataset. Do not use its metrics in a dissertation.
