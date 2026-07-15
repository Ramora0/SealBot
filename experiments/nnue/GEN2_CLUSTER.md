# Gen2 self-play generation — cluster handoff (OSC Pitzer, CPU nodes)

Goal: mass self-play of the mixnet engine with own-search root-score
labels + outcomes — the gen2 corpus for the Stockfish-style loop.
Target: 100–300k games ≈ 3–10M positions. datagen.py is resumable
(per-worker shard files); kill/requeue is safe.

## One-time setup on the cluster

    cd ~/personal/SealBot && git fetch && git checkout mixnet-repro
    # build the mixnet engine (icpc, force-clean):
    python autoresearch/build.py cand_mixnet --smoke
    # bake the blob from the committed checkpoint — MUST be --device cpu
    # (CPU bake is the parity-verified canonical table; CUDA bake drifts):
    cd experiments/strix
    python mixnet_bake.py --ckpt output_mixnet1m/mixnet.pt --device cpu \
        --out ../../cand_mixnet/mixnet.bin

## Per-job env (workers inherit; SEAL_THREADS=1 — parallelism comes
## from workers, one core each)

    export SEAL_EVAL=mixnet
    export SEAL_MIXNET_BLOB=$HOME/personal/SealBot/cand_mixnet/mixnet.bin
    export SEAL_TRUNK_BLEND=0 SEAL_POLICY_MODE=74
    export SEAL_VCF=11 SEAL_VCF_K=9 SEAL_VCF_BUDGET=25000
    export SEAL_SMP_MODE=2 SEAL_THREADS=1

## Run (40-core node: 38 workers, leave 2 for OS)

    cd ~/personal/SealBot/experiments/nnue
    python datagen.py --bot-dir ../../cand_mixnet --out data/gen2 \
        --workers 38 --games 150000 \
        --tl-min 0.08 --tl-max 0.16 --open-min 2 --open-max 12

Notes:
- tl 0.08–0.16 (vs gen0's 0.02–0.05): deeper root-score labels — the
  gen1 postmortem named shallow labels as a failure cause. ~30 turns
  × ~0.12 s ≈ 4 s/game/worker → ~35k games/hour/node.
- Randomized tl + random 2–12 stone openings = the diversity knob
  (Stockfish random_multi_pv analog).
- Positions record (cells, mover, moves_left, move_count, score=root
  search score mover-POV, depth) + per-game winner. This is exactly
  the trunk-era pkl format; every existing relabel/label tool applies.
- Optional on-cluster post-pass (V100 nodes): strix forward sidecars
  via strix_relabel.py (3.5k pos/s) and VCF proof flags via
  vcf_label.py — both feed the loss-level mix at train time.
- Ship back: rsync data/gen2/*.pkl (and any sidecars) to the dev box,
  or train on-cluster; the trainer only needs the hexo venv + 1 GPU.
