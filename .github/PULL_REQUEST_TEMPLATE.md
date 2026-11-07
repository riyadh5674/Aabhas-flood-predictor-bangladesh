## Summary

<!-- What does this PR change and why? -->

## Area

- [ ] API / inference
- [ ] Web app / UI
- [ ] Data ingestion / features
- [ ] Model or evaluation
- [ ] Alerts / governance
- [ ] Deployment / CI
- [ ] Documentation

## Verification

<!-- Paste the commands you ran and their results. -->

```powershell
.\.venv\Scripts\python.exe tests\run_tests.py
.\.venv\Scripts\python.exe scripts\freeze_results.py --check

cd system\web
npm run typecheck
npm run test
npm run build
```

- [ ] Python tests pass
- [ ] Freeze check unchanged (or an intentional revision is explained below)
- [ ] Frontend typecheck, tests and build pass
- [ ] No new hard-coded machine paths
- [ ] No label-window features, no test-split reads

## Scientific impact

<!-- Does this touch labels, splits, thresholds, calibration or the frozen model? If yes, explain. -->

## Screenshots (UI changes)

<!-- Before / after if possible. -->
