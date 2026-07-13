"""Bridge: SealBot positions -> hexo-strix GameStates -> batched GPU values.

Value convention: strix's value head outputs a scalar in [-1, 1], the
expected outcome from the SIDE-TO-MOVE's perspective — same POV as our
shards' mover-relative scores.

Mapping: SealBot player A -> P1, B -> P2. HeXO seeds (0,0)=P1; our datagen
always opens at (0,0) with player A, so positions map directly. If (0,0)
is missing or B-owned (defensive), the position is translated so an
A-stone lands on the origin (both evals are translation invariant).

Run inside the hexo-strix venv (python 3.13).
"""

from pathlib import Path

STRIX_ROOT = Path("/users/PAS2836/leedavis/personal/hexo-strix")
CKPT = STRIX_ROOT / "checkpoint_00237000.pt"


def load_strix(device="cuda:0"):
    import torch
    import hexo_rs
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mc = ModelConfig(**ck["model_config"])
    model = HeXONet(mc).to(device)
    sd = {k.removeprefix("_orig_mod."): v
          for k, v in ck["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    model.eval()
    gc = hexo_rs.GameConfig(**ck["game_config"])
    return model, mc, gc


def state_from_cells(cells, mover, moves_left, gc):
    """cells: [(q, r, 1|2)] absolute (1=A, 2=B); mover: 1|2; moves_left: 1|2.

    Returns a hexo_rs.GameState, or None if the position is unrepresentable
    (terminal, or A has no stones).
    """
    import hexo_rs

    owner = {(q, r): p for q, r, p in cells}
    if owner.get((0, 0)) != 1:
        a_stones = [(q, r) for q, r, p in cells if p == 1]
        if not a_stones:
            return None
        oq, orr = a_stones[0]
        owner = {(q - oq, r - orr): p for (q, r), p in owner.items()}
    stones = [((q, r), "P1" if p == 1 else "P2") for (q, r), p in owner.items()]
    gs = hexo_rs.GameState.from_state(
        stones, "P1" if mover == 1 else "P2", moves_left, gc)
    if gs.is_terminal():
        return None
    return gs


def value_batch(model, mc, states, device="cuda:0", chunk=512):
    """Single-forward strix values (side-to-move POV) for a list of states."""
    import torch
    from hexo_a0.graph import axis_states_to_batch

    out = []
    with torch.inference_mode():
        for i in range(0, len(states), chunk):
            batch, aux = axis_states_to_batch(
                states[i:i + chunk],
                prune_empty_edges=mc.prune_empty_edges,
                threat_features=mc.threat_features,
                relative_stones=mc.relative_stone_encoding,
                device=device,
            )
            _, _, values = model._forward_batch_core(
                batch,
                legal_idx=aux.legal_idx,
                stone_idx=aux.stone_idx,
                stone_batch=aux.stone_batch,
            )
            out.extend(values.float().cpu().tolist())
    return out
