# Official evaluation notebook

Open `01_Official_Evaluation.ipynb` in Jupyter and choose the Possession Lens kernel. Run all cells for the official published-baseline example. To score one frozen candidate, save its CSV beside the notebook in this `evaluation` folder. A single CSV is selected automatically. To choose manually, enter its name in `CSV_FILENAME = "my_team.csv"` and run all cells. Several CSVs require an explicit filename. No CSV and a blank field runs the baseline example.

The notebook executes the unchanged official validator and evaluator, using a pinned public reference. Public scores cover only five minutes and must not be used for training, parameter tuning or model selection. It never opens video or uploads data.

All generated results, copied candidate CSVs and logs stay in ignored `local/evaluation/<run>/`. Source hashes and the upstream commit are in `official/SOURCE.json`; official licensing is in `official/LICENSE`. Keep outputs cleared before sharing the notebook.

From the project root:

```sh
python -m pip install -r requirements-notebook.txt
jupyter lab evaluation/01_Official_Evaluation.ipynb
```

The `official/` subfolder is an unchanged snapshot of the organisers’ public code, public reference and published baseline. It does not contain licensed match tracking/video or the private reference.
