# Running an evaluation

Each agent reviews one snippet with numbered lines, a common prompt and an output schema. Labels, paired examples and previous results are withheld. Answers use the displayed line numbers.

Download and extract the dataset first. Prepare inputs and check that annotation clues have been removed:

```bash
python3 eval/render_inputs.py --output-dir results/benchmark/rendered --fresh
python3 eval/validate_inputs.py results/benchmark/rendered --schema eval/prediction_schema.json
python3 eval/create_sample_workspace.py --run benchmark --fresh --write-index
```

Set up the agent tools and credentials, then select models in `eval/agent_matrix.yaml`. Check the commands without calling models:

```bash
python3 eval/run_agent_matrix.py --run benchmark --mode smoke --dry-run
```

Run the evaluation, then check that all samples were processed with the intended settings:

```bash
python3 eval/run_agent_matrix.py --run benchmark --mode full
python3 eval/validate_runs.py --run benchmark
python3 eval/analysis.py --run benchmark
```

Model calls may incur provider charges. Use a separate checkout for new evaluations to preserve the downloaded reference results. DeepSeek credentials are read from `DEEPSEEK_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an explicitly selected `DEEPSEEK_ENV_FILE`.
