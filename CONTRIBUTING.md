# Contributing to dsgbr

Contributions are genuinely welcome, and that includes the ones that are not
code. A bug report, a confusing docstring, a README paragraph that turned out to
be wrong, a question that took you an hour to answer yourself — all of those are
worth opening an [issue](https://github.com/openfluids/dsgbr/issues) for.

If you are unsure whether something is worth reporting, it probably is. Open the
issue.

## Getting set up

```bash
git clone https://github.com/openfluids/dsgbr.git
cd dsgbr
uv pip install -e ".[qa,tests]"
pre-commit install
```

`pre-commit install` is worth the ten seconds — it runs the same hooks CI does,
so problems surface before you push rather than after.

## Before you open a pull request

The same checks CI runs:

```bash
pre-commit run --all-files
pytest --cov=dsgbr --cov-report=term-missing
```

Some tests need extra optional dependencies:

```bash
uv pip install -e ".[tests,tests-extra]"
```

If one fails for a reason you think is unrelated to your change, say so in the
pull request rather than working around it — that is useful information, and
sometimes it is CI that is wrong.

## What makes a pull request easy to review

- **One thing at a time.** A focused change gets reviewed quickly. A change that
  also reformats fifty unrelated lines is hard to read and slow to merge.
- **Say what you verified.** A pasted command and its output is worth more than
  "tested locally".
- **Ask early.** For anything substantial, open an issue first. It is much
  better to disagree about an approach before you have written it than after.
- **Draft PRs are fine.** Opening one early to ask "is this the right
  direction?" is welcome and costs nothing.

Reviews may take a few days — one maintainer, research alongside. A nudge on a
quiet pull request is welcome, not annoying.

## Conventions

Only the ones that are actually enforced:

- Spectral estimates need a test against a signal with a known answer, not only
  a regression against current output. A test that would pass if the estimator
  were subtly wrong is not much of a test.
- Use `np.random.default_rng(seed)`, not the legacy module-level `np.random`
  calls — reproducibility matters more here than convenience.
- Resolve paths relative to `__file__`. No hardcoded absolute paths.
- Formatting, import order and the rest are handled by `pre-commit` — do not
  hand-tune them.
- Documentation links are checked in CI; if you add one, make sure it resolves.

## Conduct and licence

Everyone taking part is asked to follow the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md).
It is short.

dsgbr is licensed under Apache-2.0, and contributions are accepted under the
same licence. See `LICENSE` and `NOTICE`.

Found a security problem? Please do not open a public issue — see the
[security policy](https://github.com/openfluids/dsgbr/security/policy).
