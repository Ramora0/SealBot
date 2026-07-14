/*
 * vcf.h -- Threat-space (VCF-style) forced-win solver for Connect6.
 *
 * Explores ONLY forcing attacker turns (turns that create four/five-threats)
 * and the forced defender replies (minimal blocking placements), so it can
 * prove deep forced wins in microseconds.
 *
 * Connect6-specific threat model (defender places 2 stones per turn):
 *   five-threat = 6-window with 5 attacker stones, 0 defender (1 empty)
 *   four-threat = 6-window with 4 attacker stones, 0 defender (2 empties)
 * After a forcing attacker turn the defender must occupy, with 2 stones,
 * at least one empty of EVERY live threat window (a hitting set):
 *   min hitting set >= 3  ->  attacker wins next turn (+1)
 *   min hitting set == 2  ->  defender replies are exactly the 2-cell covers;
 *                             recurse on each (all must lose for +1)
 *   min hitting set <= 1  ->  defender keeps a free stone; out of scope for
 *                             v1 (cannot enumerate "anywhere"), return unknown
 *
 * v1 soundness rule for defender counter-attacks: at every attacker node,
 * if the defender has an immediate win of their own (a clean 4/5-window
 * they could complete on their next turn), return 0 (unknown).  Attacker
 * stones never help defender windows, so this check (done before the
 * attacker's turn) also covers the position after the attacker's stones.
 *
 * Return contract (forced_win):
 *   +1  proven forced win within max_turns attacker turns (writes first turn)
 *   -1  provably no forced win via this threat-space model within max_turns
 *    0  unknown (depth/budget cutoff, defender counter-threat, free-stone
 *       defense, or capped enumeration somewhere in the tree)
 * +1 is conservative: every enumerated defender reply must lose.
 */
#pragma once

#include "bot.h"

namespace opt {

static constexpr int VCF_MAX_THREAT_WINDOWS = 32;
static constexpr int VCF_MAX_U              = 24;
static constexpr int VCF_MAX_CANDS          = 192;

struct MinimaxBot::VcfState {
    flat_map<uint64_t, int8_t> memo;   // (hash ^ stones ^ remaining) -> result
    long    nodes      = 0;
    long    budget     = 5000;     // overwritten from vcf_node_budget
    bool    incomplete = false;        // any pruned/unknown branch anywhere
    int32_t gen        = 0;            // generation counter for stamp arrays
    int32_t win_stamp[3][ARR][ARR] = {};
    int32_t cell_stamp[ARR][ARR]   = {};
    int32_t cell_idx[ARR][ARR]     = {};
    int8_t  attacker = P_A, defender = P_B;
};

// ────────────────────────────────────────────────────────────────
//  Position loading (mirror of get_move's init, minus eval windows)
// ────────────────────────────────────────────────────────────────
inline void MinimaxBot::_vcf_load_position(const GameState& gs) {
    std::memset(_board, 0, sizeof(_board));
    std::memset(_wc, 0, sizeof(_wc));
    std::memset(_wp, 0, sizeof(_wp));
    std::memset(_cand_rc, 0, sizeof(_cand_rc));
    _board_cells.clear();
    _hot_a.clear();
    _hot_b.clear();
    _cand_set.clear();
    _rc_stack.clear();

    for (const auto& cell : gs.cells) {
        _board[cell.q + OFF][cell.r + OFF] = cell.player;
        _board_cells.push_back(pack(cell.q, cell.r));
    }

    _cur_player = gs.cur_player;
    _moves_left = gs.moves_left;
    _move_count = gs.move_count;
    _winner     = P_NONE;
    _game_over  = false;
    _player     = gs.cur_player;
    _ply        = 0;
    _eval_score = 0.0;   // _wp zeroed above: eval deltas stay in-bounds, unused

    if (_player == P_A) { _cell_a = 1; _cell_b = 2; }
    else                { _cell_a = 2; _cell_b = 1; }

    _hash = 0;
    for (Coord c : _board_cells)
        _hash ^= get_zobrist(pack_q(c), pack_r(c),
                             _board[pack_q(c) + OFF][pack_r(c) + OFF]);

    // 6-cell windows + hot sets
    for (Coord c : _board_cells) {
        int bq = pack_q(c), br = pack_r(c);
        int bqi = bq + OFF, bri = br + OFF;
        for (const auto& wo : g_win_offsets) {
            int sqi = bqi - wo.oq, sri = bri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            if (counts.first != 0 || counts.second != 0) continue;
            int d = wo.d_idx;
            int sq = bq - wo.oq, sr = br - wo.or_;
            int ac = 0, bc = 0;
            for (int j = 0; j < WIN_LENGTH; j++) {
                int8_t v = _board[sq + j * DIR_Q[d] + OFF][sr + j * DIR_R[d] + OFF];
                if (v == P_A) ac++;
                else if (v == P_B) bc++;
            }
            if (ac || bc) {
                counts = {static_cast<int8_t>(ac), static_cast<int8_t>(bc)};
                if (ac >= 4) _hot_a.insert(wo.d_idx, sqi, sri);
                if (bc >= 4) _hot_b.insert(wo.d_idx, sqi, sri);
            }
        }
    }

    // Candidates
    for (Coord c : _board_cells) {
        int bq = pack_q(c), br = pack_r(c);
        for (const auto& nb : g_nb_offsets) {
            int nq = bq + nb.dq, nr = br + nb.dr;
            int nqi = nq + OFF, nri = nr + OFF;
            _cand_rc[nqi][nri]++;
            if (_board[nqi][nri] == 0)
                _cand_set.insert(pack(nq, nr));
        }
    }
}

// ────────────────────────────────────────────────────────────────
//  Defender node: after a (candidate) attacker turn was made.
//  Returns  1 win proven, 0 unknown/refuted, -2 turn was non-forcing.
// ────────────────────────────────────────────────────────────────
inline int MinimaxBot::_vcf_after_attack_turn(int remaining) {
    VcfState& V = *_vcf;
    if (++V.nodes > V.budget) { V.incomplete = true; return 0; }

    const bool att_is_a = (V.attacker == P_A);
    const auto& hot = att_is_a ? _hot_a : _hot_b;

    // Live attacker threat windows (4 or 5 attacker stones, 0 defender).
    Coord we[VCF_MAX_THREAT_WINDOWS][2];
    int   wn[VCF_MAX_THREAT_WINDOWS];
    int   ntw = 0;
    for (const auto& he : hot.vec) {
        const auto& cc = _wc[he.d][he.qi][he.ri];
        int my  = att_is_a ? cc.first  : cc.second;
        int opp = att_is_a ? cc.second : cc.first;
        if (opp != 0 || my < 4) continue;
        if (ntw >= VCF_MAX_THREAT_WINDOWS) { V.incomplete = true; return 0; }
        int sq = he.qi - OFF, sr = he.ri - OFF;
        int n = 0;
        for (int j = 0; j < WIN_LENGTH; j++) {
            int cq = sq + j * DIR_Q[he.d], cr = sr + j * DIR_R[he.d];
            if (_board[cq + OFF][cr + OFF] == 0)
                we[ntw][n++] = pack(cq, cr);
        }
        wn[ntw] = n;   // 1 (five-threat) or 2 (four-threat)
        ntw++;
    }
    if (ntw == 0) return -2;   // no threat created: not a forcing turn

    // Union of blocking cells.
    Coord U[VCF_MAX_U];
    int nu = 0;
    const int32_t g_u = ++V.gen;
    for (int w = 0; w < ntw; w++)
        for (int k = 0; k < wn[w]; k++) {
            Coord c = we[w][k];
            int qi = pack_q(c) + OFF, ri = pack_r(c) + OFF;
            if (V.cell_stamp[qi][ri] == g_u) continue;
            V.cell_stamp[qi][ri] = g_u;
            if (nu >= VCF_MAX_U) { V.incomplete = true; return 0; }
            U[nu++] = c;
        }

    auto hits = [&](int w, Coord a, Coord b) {
        return we[w][0] == a || we[w][0] == b ||
               (wn[w] > 1 && (we[w][1] == a || we[w][1] == b));
    };

    // Single-stone cover -> defender keeps a free stone; unsound to restrict
    // where it goes, so this line is unknown in v1.
    for (int u = 0; u < nu; u++) {
        bool all = true;
        for (int w = 0; w < ntw; w++)
            if (!hits(w, U[u], U[u])) { all = false; break; }
        if (all) { V.incomplete = true; return 0; }
    }

    // Enumerate ALL 2-stone covers: these are exactly the defender replies
    // that avoid immediate loss (any non-cover leaves a clean 4/5-window and
    // the attacker completes 6 next turn; defender cannot win first -- the
    // caller verified the defender had no immediate win, and attacker stones
    // never enable defender windows).
    Turn reps[VCF_MAX_U * (VCF_MAX_U - 1) / 2];
    int nrep = 0;
    for (int i = 0; i < nu; i++)
        for (int j = i + 1; j < nu; j++) {
            bool all = true;
            for (int w = 0; w < ntw; w++)
                if (!hits(w, U[i], U[j])) { all = false; break; }
            if (all)
                reps[nrep++] = {coord_min(U[i], U[j]), coord_max(U[i], U[j])};
        }

    if (nrep == 0) return 1;   // min hitting set >= 3: unstoppable

    for (int i = 0; i < nrep; i++) {
        UndoStep steps[2];
        int n = _make_turn(reps[i], steps);
        int r;
        if (_game_over)        // defender completed 6 while blocking (cannot
            r = 0;             // happen given the pre-check; be safe anyway)
        else
            r = _vcf_attack(2, remaining - 1, nullptr);
        _undo_turn(steps, n);
        if (r != 1) return 0;  // this defense survives -> not a proven win
    }
    return 1;                  // every legal defense loses
}

// ────────────────────────────────────────────────────────────────
//  Attacker node.
// ────────────────────────────────────────────────────────────────
inline int MinimaxBot::_vcf_attack(int stones, int remaining, Turn* out) {
    VcfState& V = *_vcf;
    if (++V.nodes > V.budget) { V.incomplete = true; return 0; }
    if (_game_over) return (_winner == V.attacker) ? 1 : 0;
    if (remaining <= 0) { V.incomplete = true; return 0; }

    const bool att_is_a = (V.attacker == P_A);

    // 1) Immediate win this turn?
    if (stones >= 2) {
        auto iw = _find_instant_win(V.attacker);
        if (iw.first) { if (out) *out = iw.second; return 1; }
    } else {
        // 1 stone left: need a five-threat (5 attacker stones, 0 defender).
        const auto& hot = att_is_a ? _hot_a : _hot_b;
        for (const auto& he : hot.vec) {
            const auto& cc = _wc[he.d][he.qi][he.ri];
            int my  = att_is_a ? cc.first  : cc.second;
            int opp = att_is_a ? cc.second : cc.first;
            if (my != WIN_LENGTH - 1 || opp != 0) continue;
            int sq = he.qi - OFF, sr = he.ri - OFF;
            for (int j = 0; j < WIN_LENGTH; j++) {
                int cq = sq + j * DIR_Q[he.d], cr = sr + j * DIR_R[he.d];
                if (_board[cq + OFF][cr + OFF] == 0) {
                    if (out) *out = {pack(cq, cr), pack(cq, cr)};
                    return 1;
                }
            }
        }
    }

    // 2) v1 soundness rule: defender with an immediate win is out of scope.
    {
        auto dw = _find_instant_win(V.defender);
        if (dw.first) { V.incomplete = true; return 0; }
    }

    // 3) Any deeper win needs a forcing turn now plus a finishing turn later.
    if (remaining < 2) { V.incomplete = true; return 0; }

    // 4) Memoization (exact subproblem: position x stones x remaining).
    //    Stored tri-state: 1 win, 0 unknown (pruned somewhere), -1 no win
    //    via forcing lines (complete). Entries depend only on (position,
    //    stones, remaining), so they stay valid across the iterative-
    //    deepening loop in forced_win.
    const uint64_t mkey = _hash
        ^ (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(stones))
        ^ (0xc2b2ae3d27d4eb4fULL * static_cast<uint64_t>(remaining));
    if (!out) {
        auto it = V.memo.find(mkey);
        if (it != V.memo.end()) {
            if (it->second == 1) return 1;
            if (it->second == 0) V.incomplete = true;
            return 0;
        }
    }

    // Track incompleteness of THIS subtree separately so the memo entry
    // can distinguish "proven no forcing win" from "unknown".
    const bool saved_inc = V.incomplete;
    V.incomplete = false;

    // 5) Threat-building structure around attacker stones.
    //    cells3: empty cells converting a clean 3-window into a four-threat
    //            (plus, mid-turn with 1 stone, a clean 4-window into a five).
    //    twos:   clean 2-windows -- a pair of their empties builds a four.
    struct TwoWin { int d, qi, ri; };
    std::vector<std::pair<int, Coord>> cells3;   // (windows completed, cell)
    std::vector<TwoWin> twos;

    const int32_t g_win = ++V.gen;
    const int32_t g_c3  = ++V.gen;
    const int32_t g_tw  = ++V.gen;

    for (Coord bc : _board_cells) {
        int bq = pack_q(bc), br = pack_r(bc);
        if (_board[bq + OFF][br + OFF] != V.attacker) continue;
        for (const auto& wo : g_win_offsets) {
            int sqi = bq + OFF - wo.oq, sri = br + OFF - wo.or_;
            if (V.win_stamp[wo.d_idx][sqi][sri] == g_win) continue;
            V.win_stamp[wo.d_idx][sqi][sri] = g_win;
            const auto& cc = _wc[wo.d_idx][sqi][sri];
            int my  = att_is_a ? cc.first  : cc.second;
            int opp = att_is_a ? cc.second : cc.first;
            if (opp != 0) continue;
            if (my == 3 || (stones == 1 && my == 4)) {
                int sq = sqi - OFF, sr = sri - OFF;
                for (int j = 0; j < WIN_LENGTH; j++) {
                    int cq = sq + j * DIR_Q[wo.d_idx];
                    int cr = sr + j * DIR_R[wo.d_idx];
                    int qi2 = cq + OFF, ri2 = cr + OFF;
                    if (_board[qi2][ri2] != 0) continue;
                    if (V.cell_stamp[qi2][ri2] == g_c3) {
                        cells3[V.cell_idx[qi2][ri2]].first++;
                    } else {
                        V.cell_stamp[qi2][ri2] = g_c3;
                        V.cell_idx[qi2][ri2] = static_cast<int32_t>(cells3.size());
                        cells3.push_back({1, pack(cq, cr)});
                    }
                }
            } else if (my == 2 && stones == 2) {
                twos.push_back({wo.d_idx, sqi, sri});
            }
        }
    }

    // Double-completion cells first: they force with a single stone.
    std::sort(cells3.begin(), cells3.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return coord_lt(a.second, b.second);
              });

    // 6) Candidate forcing turns.
    std::vector<Turn> cands;
    bool truncated = false;
    flat_set<Turn, TurnHash> seen;
    auto push_turn = [&](Coord a, Coord b) {
        if (static_cast<int>(cands.size()) >= VCF_MAX_CANDS) {
            truncated = true;
            return;
        }
        Turn t{coord_min(a, b), coord_max(a, b)};
        if (seen.emplace(t).second) cands.push_back(t);
    };

    if (stones == 1) {
        for (const auto& [cnt, c] : cells3)
            cands.push_back({c, c});
    } else {
        const int n3 = static_cast<int>(cells3.size());

        // T1: both stones complete a clean 3-window each.
        for (int i = 0; i < n3; i++)
            for (int j = i + 1; j < n3; j++)
                push_turn(cells3[i].second, cells3[j].second);

        // Deduped empties of clean 2-windows (builder cells).
        std::vector<Coord> twos_empties;
        for (const auto& tw : twos) {
            int sq = tw.qi - OFF, sr = tw.ri - OFF;
            for (int j = 0; j < WIN_LENGTH; j++) {
                int cq = sq + j * DIR_Q[tw.d], cr = sr + j * DIR_R[tw.d];
                int qi2 = cq + OFF, ri2 = cr + OFF;
                if (_board[qi2][ri2] != 0) continue;
                if (V.cell_stamp[qi2][ri2] == g_c3) continue;  // already a completer
                if (V.cell_stamp[qi2][ri2] == g_tw) continue;
                V.cell_stamp[qi2][ri2] = g_tw;
                twos_empties.push_back(pack(cq, cr));
            }
        }

        // T2: pair of empties jointly completing a clean 2-window; kept if a
        // stone also completes a 3-window, or the pair upgrades >= 2 windows.
        for (const auto& tw : twos) {
            const int d = tw.d;
            int sq = tw.qi - OFF, sr = tw.ri - OFF;
            Coord em[6];
            int   idx[6], ne = 0;
            for (int j = 0; j < WIN_LENGTH; j++) {
                int cq = sq + j * DIR_Q[d], cr = sr + j * DIR_R[d];
                if (_board[cq + OFF][cr + OFF] == 0) {
                    em[ne] = pack(cq, cr);
                    idx[ne] = j;
                    ne++;
                }
            }
            for (int i = 0; i < ne; i++)
                for (int j = i + 1; j < ne; j++) {
                    bool c3a = V.cell_stamp[pack_q(em[i]) + OFF][pack_r(em[i]) + OFF] == g_c3;
                    bool c3b = V.cell_stamp[pack_q(em[j]) + OFF][pack_r(em[j]) + OFF] == g_c3;
                    if (!c3a && !c3b) {
                        // Count clean 2-windows along d containing both cells.
                        int gap = idx[j] - idx[i];
                        int q1 = pack_q(em[i]), r1 = pack_r(em[i]);
                        int cnt = 0;
                        for (int m = 0; m + gap <= WIN_LENGTH - 1; m++) {
                            int aq = q1 - m * DIR_Q[d], ar = r1 - m * DIR_R[d];
                            const auto& c2 = _wc[d][aq + OFF][ar + OFF];
                            int my2  = att_is_a ? c2.first  : c2.second;
                            int opp2 = att_is_a ? c2.second : c2.first;
                            if (my2 == 2 && opp2 == 0) cnt++;
                        }
                        if (cnt < 2) continue;
                    }
                    push_turn(em[i], em[j]);
                }
        }

        // T1c: double-completion cell (forces alone) + a builder stone.
        for (const auto& [cnt, c] : cells3) {
            if (cnt < 2) break;   // sorted descending
            for (Coord y : twos_empties)
                push_turn(c, y);
        }
        // Fallback: lone double-maker with no 2-window structure nearby.
        if (cands.empty() && n3 > 0 && cells3[0].first >= 2) {
            int added = 0;
            for (Coord y : _cand_set) {
                if (y == cells3[0].second) continue;
                push_turn(cells3[0].second, y);
                if (++added >= 4) break;
            }
        }
    }
    if (truncated) V.incomplete = true;

    // 7) Try each candidate: make it, require it to be forcing, and require
    //    every enumerated defender cover to lose.
    int result = 0;
    for (const Turn& t : cands) {
        if (V.nodes > V.budget) { V.incomplete = true; break; }
        int r;
        if (stones == 1) {
            int q = pack_q(t.first), rr = pack_r(t.first);
            SavedState st{_cur_player, _moves_left, _winner, _game_over};
            int8_t pl = _cur_player;
            _make(q, rr);
            r = _game_over ? ((_winner == V.attacker) ? 1 : 0)
                           : _vcf_after_attack_turn(remaining);
            _undo(q, rr, st, pl);
        } else {
            UndoStep steps[2];
            int n = _make_turn(t, steps);
            r = _game_over ? ((_winner == V.attacker) ? 1 : 0)
                           : _vcf_after_attack_turn(remaining);
            _undo_turn(steps, n);
        }
        if (r == 1) {
            result = 1;
            if (out) *out = t;
            break;
        }
        // r == 0 (unknown/refuted) or -2 (non-forcing): try the next one.
    }

    const bool node_inc = V.incomplete;
    V.incomplete = saved_inc || node_inc;
    V.memo[mkey] = static_cast<int8_t>(result == 1 ? 1 : (node_inc ? 0 : -1));
    return result;
}

// ────────────────────────────────────────────────────────────────
//  Public entry points.
// ────────────────────────────────────────────────────────────────
inline int MinimaxBot::forced_win(int8_t player, int stones_left,
                                  int max_turns, Turn* out) {
    if (!_vcf) _vcf = std::make_unique<VcfState>();
    VcfState& V = *_vcf;
    V.memo.clear();
    V.nodes      = 0;
    V.budget     = vcf_node_budget;
    V.incomplete = false;
    V.attacker   = player;
    V.defender   = (player == P_A) ? P_B : P_A;
    if (V.gen > 2000000000) {   // stamp-generation wrap guard
        std::memset(V.win_stamp, 0, sizeof(V.win_stamp));
        std::memset(V.cell_stamp, 0, sizeof(V.cell_stamp));
        V.gen = 0;
    }

    vcf_nodes = 0;
    if (_game_over || player != _cur_player || stones_left != _moves_left)
        return 0;

    // Iterative deepening: cheap shallow proofs first (also yields the
    // shortest win found), one shared node budget, memo reused across
    // iterations (entries are keyed by `remaining`, so they stay exact).
    Turn t{};
    int  r = 0;
    bool last_inc = false;
    for (int k = 1; k <= max_turns; k++) {
        V.incomplete = false;
        r = _vcf_attack(stones_left, k, &t);
        last_inc = V.incomplete;
        if (r == 1) break;
        if (V.nodes > V.budget) { last_inc = true; break; }
        // Complete no-win at horizon k: depth was never the binding
        // constraint, so no deeper horizon can help either.
        if (!last_inc) break;
    }
    vcf_nodes = static_cast<int>(V.nodes);
    if (r == 1) {
        if (out) *out = t;
        return 1;
    }
    return last_inc ? 0 : -1;
}

inline int MinimaxBot::forced_win_position(const GameState& gs, int max_turns,
                                           Turn* out) {
    _vcf_load_position(gs);
    return forced_win(gs.cur_player, gs.moves_left, max_turns, out);
}

} // namespace opt
