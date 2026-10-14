# MilitaryOR-100

A benchmark and a formal-reasoning agent for machine-checkable verification of military decision-making.

MilitaryOR-100 pairs 100 military operations research problems (air-defense weapon--target assignment) with reference Lean 4 formalizations. It is used to evaluate an LLM-based formal reasoning agent that performs **autoformalization**, **theorem proving**, and **Lean 4 verification**, with verifier-guided self-correction and proof-or-refutation reasoning.

## Dataset

| File | Description |
|---|---|
| `data/ADFP_100.jsonl` | The full 100-problem benchmark (natural language + reference Lean statements). |
| `data/ADFP_87_verified.jsonl` | The 87 problems whose autoformalized Lean statements passed type checking (adds the `lean4_statement` field). |

Each record contains: `id`, `domain`, `task_type`, `difficulty`, `focus`, `synthetic`, `problem_text`, `objective`, `constraints`, `assumptions`, `params`, and `lean_target`.

## Code

| File | Role |
|---|---|
| `src/modeling.py` | Modeling layer: natural language -> Lean 4 statement (autoformalization). |
| `src/solver.py` | Solving layer: Lean statement -> proof (verifier-guided self-correction). |
| `src/lean4_verifier.py`, `src/verify_api.py` | Lean 4 verification via a Kimina Lean server. |
| `src/start_kimina.py` | Launches the Kimina Lean 4 server. |
| `src/pipeline.py` | Main entry point (modeling -> verification -> solving). |
| `src/generate_report.py` | Evaluation report generation. |
| `src/config.py` | Global configuration (model paths, sampling, GPU, timeouts). |
| `src/lean4_utils.py`, `src/solver_worker.py` | Lean helpers and parallel solving worker. |

## Requirements

- Python 3.9+
- Lean 4 with a running Kimina Lean server
- Model checkpoints: ATF-8B (modeling layer) and Goedel-Prover-V2-8B (solving layer)

## Usage

1. Set the model paths in `src/config.py` (`MODELING_MODEL_DIR`, `SOLVER_MODEL_PATH`).
2. Start the Lean 4 verifier: `python src/start_kimina.py`.
3. Run the pipeline: `python src/pipeline.py --input-file data/ADFP_100.jsonl`.
4. Generate the report: `python src/generate_report.py`.

## License

- Source code: MIT (see `LICENSE`).
- Dataset: CC BY 4.0.

## Ethics and dual-use notice

MilitaryOR-100 consists of anonymized, desensitized, and semi-synthetic scenarios constructed solely for academic research on the correctness and verifiability of LLM reasoning. It is intended strictly for research and reliability evaluation, and must not be used for operational military deployment, tactical planning, or kinetic actions.

## Citation

If you use this dataset or code, please cite the corresponding paper (author list to be completed):

```bibtex
@article{militaryor100,
  title   = {Towards Verifiable Military Decision-Making through Formal Reasoning},
  author  = {TBD},
  journal = {Defence Technology},
  year    = {2026}
}
```
