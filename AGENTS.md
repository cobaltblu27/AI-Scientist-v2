# AGENTS.md

* This machine uses 32Gb RAM, and 3070 8g.
* Using micromamba environment ai_scientist.
* On zshrc, the conda is aliased to micromamba. there might be jobs that work on conda but not in mamba, so keep in mind this project is built using mamba.
* User is operating on conda activated environment, so when running it yourself make sure inline it with micromamba like:
```sh
micromamba run -n ai_scientist python ai_scientist/perform_ideation_temp_free.py --workshop-file ai_scientist/ideas/drp_research.md --model copilot-gpt-4o --max-num-generations 3 --num-reflections 3
```

* However, when giving instruction to user, remember user has conda (micromamba) activated, it is okay to give instruction without `micromamba run -n`.

## Behavioral rules

* If requirements are ambiguous: stop and ask before coding.
* Prefer minimal diffs; avoid refactors unless required by acceptance criteria.
* When asked to summarize experiment results, also update `RESULTS.md` with per-experiment and per-attempt outcomes, key metrics, and notable artifacts.

## Contribution
this is forked repository, and development is mainly for my own use.
when creating pr, point to main branch of remote `cobalt`.
you may use gh.
