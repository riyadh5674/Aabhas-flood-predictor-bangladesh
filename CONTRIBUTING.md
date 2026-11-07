# Contributing to Aabhas

Thanks for taking the time to improve flood forecasting for Bangladesh. This project is
a research prototype, so the bar is **reproducibility and honesty**, not just code that
runs.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

cd system\web
npm ci
cd ..\..

# full stack
cd system\web
npm run dev
```

## Before you open a pull request

```powershell
# 1. Python tests — must stay green (84 currently)
.\.venv\Scripts\python.exe tests\run_tests.py

# 2. Frozen evidence — must not change without an intentional revision
.\.venv\Scripts\python.exe scripts\freeze_results.py --check

# 3. Frontend
cd system\web
npm run typecheck
npm run test
npm run build
```

If your change intentionally updates a frozen artefact, say so in the PR description,
update `reports/freeze_manifest.json` through the freeze script, and explain why.

## What we welcome

- Bug fixes and test coverage for the API, pipeline, alert engine or UI.
- Documentation improvements, especially Bengali copy review.
- Accessibility and low-bandwidth/offline improvements.
- Data-source reliability work (retry, freshness, schema checks).
- Evaluation and reproducibility improvements that do not re-open the test split.

## What needs a discussion first

Open an issue before working on:

- changes to labels, splits, thresholds or the frozen model;
- any new external data source with licensing implications;
- changes to alert delivery or public-warning behaviour;
- large dependency or architecture changes.

## Scientific rules

These are non-negotiable:

1. **Do not re-open the test split.** If a script must read it, log the read in
   `reports/test_read_ledger.json` and get maintainer approval.
2. **Never use label-window columns as features.** `obs_days_*` and label columns are
   banned; tests assert their absence.
3. **Report per-cell results.** Never pool results across stations or periods to make a
   headline number look better.
4. **Keep missing values missing.** Do not silently fill labels or flatten NaN to `0`.
5. **Document negative results.** A failed experiment is a valid outcome.

## Style

- Python: type hints where practical, path handling via `AABHAS_ROOT`/file-relative
  resolution, no hard-coded machine paths.
- TypeScript: strict mode, no `any` unless justified, keep components small.
- Commit messages: imperative, scoped (`api: reject empty batch request`).
- Keep PRs focused. One concern per PR is easier to review and easier to reproduce.

## Reporting bugs

Use the GitHub issue templates. For security issues, follow [SECURITY.md](SECURITY.md)
and do **not** open a public issue.

## Code of conduct

By participating you agree to [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Contributions are accepted under the MIT License (see [LICENSE](LICENSE)).
