# CARVE

CARVE is a training-free, black-box counterfactual probe for auditing
tool-using video agents.

Given a frozen agent trajectory, CARVE compares two matched replay
conditions:

- **SHAM:** rerun the answering pipeline with the original retrieved
  visual evidence.
- **DESTROY:** rerun the same pipeline after Fourier phase-randomizing
  that evidence.

CARVE measures the difference between the resulting answer-change rates:

$$ \widehat{\Delta}=\widehat{\delta}_{\mathrm{destroy}}-\widehat{\delta}_{\mathrm{sham}} $$

The score measures counterfactual sensitivity to retrieved evidence
beyond the variability introduced by replaying the pipeline itself.

CARVE should be treated as a budget-dependent auditing and routing
signal. It is not a direct certificate of correctness, grounding, or
repairability.

The accompanying paper is currently under anonymous review. A public
paper link and citation information will be added after release.

## Repository structure

```text
carve/          Core CARVE probe and controller implementation
baselines/      Comparison and fallback-selection methods
configs/        Experiment and policy configuration files
experiments/    Experiment entry points
integrations/   Integration code for the evaluated video-agent scaffold
paper/          Scripts used to generate reported tables and analyses
artifacts/      Compact processed outputs used in the paper
data/           Documentation for externally hosted datasets and outputs
docs/           Reproduction instructions
tests/          Regression and unit tests
```

The core CARVE implementation is contained in `carve/` and is separated
from the evaluated agent integration and experiment-specific code.

## Installation

Create a Python environment and install the repository in editable mode:

```bash
git clone https://github.com/ramaalhamidi/CARVE.git
cd CARVE

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
git clone https://github.com/ramaalhamidi/CARVE.git
cd CARVE

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Install the test dependencies with:

```bash
pip install pytest
```

## Tests

Run:

```bash
pytest
```

Pytest is configured to collect tests only from the `tests/` directory.

## Reproducing the paper results

See:

```text
docs/REPRODUCING.md
```

The reproduction guide maps the reported experiments and tables to their
corresponding scripts, configurations, and processed artifacts.

Large datasets, model checkpoints, videos, and complete raw trajectories
are not stored directly in this repository. Their expected locations and
acquisition instructions are documented in:

```text
data/EXTERNAL_DATA.md
```

## Video-agent integration

CARVE was evaluated using a VideoExplorer-style tool-using video agent.

The upstream agent repository is not redistributed as part of this
repository. Integration instructions and CARVE-specific hooks are located
under:

```text
integrations/videoexplorer/
```

Users must obtain the upstream implementation and model dependencies
separately, subject to their original licenses and terms.

## Scope of the released artifacts

The repository includes compact processed records needed to verify the
reported analyses and regression tests.

It does not include:

- model checkpoints;
- benchmark videos;
- full raw agent trajectories;
- private credentials or service configuration;
- the complete upstream video-agent source tree.

## License

A repository license will be added before the final public release.

Third-party models, datasets, and agent implementations remain subject to
their respective licenses and terms.

## Citation

Citation information will be added after the paper is publicly released.
