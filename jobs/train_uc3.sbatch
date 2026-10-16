#!/bin/bash
# CucuGAN-Training auf dem bwUniCluster 3.0 (Slurm).
#
# Einmalig vorher: Repo klonen, .venv anlegen, Workspace "cucugan" mit collage128/ darin,
# und "mkdir -p logs" im Repo-Ordner (sonst bricht Slurm ohne Meldung ab).
#
# Abschicken (aus dem Repo-Ordner), Einstellungen über Umgebungsvariablen:
#   RUN=collage66_struct10_att EXTRA="--amp --lambda-struct 10 --cycle-noise 0.1 --attention" \
#       sbatch jobs/train_uc3.sbatch
#   Kurztest:  ... sbatch --partition=dev_gpu_h100 --time=00:30:00 jobs/train_uc3.sbatch
#   Fortsetzen: EXTRA="... --resume" (latest.pt liegt im Workspace)
# Beobachten:  squeue    |  tail -f logs/<name>_<jobid>.out    |  abbrechen: scancel <jobid>
#
#SBATCH --job-name=cucugan
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

RUN=${RUN:?Bitte RUN=<name> setzen}
EXTRA=${EXTRA:-}
WS=$(ws_find cucugan)

# Daten auf die schnelle lokale SSD des Knotens kopieren ($TMPDIR, nur während des Jobs da)
cp -r "$WS/collage128" "$TMPDIR/"

# Checkpoints und Ausgaben im Workspace ablegen (nicht im Home mit Platzbeschränkung)
mkdir -p "$WS/checkpoints" "$WS/outputs"
ln -sfn "$WS/checkpoints" checkpoints
ln -sfn "$WS/outputs" outputs

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Lauf: $RUN | Optionen: $EXTRA"
python src/train.py --name "$RUN" --data "$TMPDIR/collage128" --num-workers 6 $EXTRA 2>&1 | grep -v "it/s]$"
