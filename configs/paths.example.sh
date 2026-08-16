# Machine-specific roots. Copy to configs/paths.sh (gitignored) and edit.
# Everything in scripts/ sources this file.
export DATA_ROOT="$HOME/dataset"        # dataset/ (raw/, processed/) — big, keep OFF small drives
export RUNS_ROOT="$PWD/runs"            # experiment outputs (gitignored)
export CONDA_ENV="gsplat"               # the env that has torch+gsplat+lpips
