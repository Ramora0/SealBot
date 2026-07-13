/*
 * board.h -- MinimaxBot board operations: make/undo, move delta, eval tables.
 */
#pragma once

#include "bot.h"

namespace opt {

// ────────────────────────────────────────────────────────────────
//  Pattern table construction
// ────────────────────────────────────────────────────────────────
inline void MinimaxBot::_build_eval_tables() {
    _eval_offsets.clear();
    for (int d = 0; d < 3; d++)
        for (int k = 0; k < _eval_length; k++)
            _eval_offsets.push_back({d, k, k * DIR_Q[d], k * DIR_R[d]});
    _pow3.resize(_eval_length);
    _pow3[0] = 1;
    for (int i = 1; i < _eval_length; i++)
        _pow3[i] = _pow3[i - 1] * 3;
}

// ────────────────────────────────────────────────────────────────
//  Incremental make / undo
// ────────────────────────────────────────────────────────────────
inline void MinimaxBot::_make(int q, int r) {
    int8_t player = _cur_player;

    // Zobrist
    _hash ^= get_zobrist(q, r, player);

    int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;
    int qi = q + OFF, ri = r + OFF;

    // ── Conjunction features: snapshot affected cell classes (pre-state) ──
    int ccq[32], ccr[32], ccls[32];
    int ncc = _collect_conj_cells(qi, ri, ccq, ccr);
    if (_need_acc2) _trunk_cells(ccq, ccr, ncc, -1.f);
    if (!_use_trunk) _conj_snapshot(ccq, ccr, ncc, ccls);

    // ── 6-cell windows ──
    bool won = false;
    if (player == P_A) {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.first++;
            if (counts.first >= 4) _hot_a.insert(wo.d_idx, sqi, sri);
            if (counts.first == WIN_LENGTH && counts.second == 0) won = true;
        }
    } else {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.second++;
            if (counts.second >= 4) _hot_b.insert(wo.d_idx, sqi, sri);
            if (counts.second == WIN_LENGTH && counts.first == 0) won = true;
        }
    }

    // ── N-cell eval windows ──
    const double* pv = _pv.data();
    for (const auto& eo : _eval_offsets) {
        int sqi = qi - eo.oq, sri = ri - eo.or_;
        int& slot = _wp[eo.d_idx][sqi][sri];
        int old_pi = slot;
        int new_pi = old_pi + cell_val * _pow3[eo.k];
        _eval_score += pv[new_pi] - pv[old_pi];
        if (_need_acc2) {
            const float* en = TRK_EW[new_pi];
            const float* eo_ = TRK_EW[old_pi];
            for (int k = 0; k < TRK_K; k++) _acc2[k] += en[k] - eo_[k];
        }
        if (!_use_trunk) {
            const float* en = NET_EW[new_pi];
            const float* eo_ = NET_EW[old_pi];
            for (int k = 0; k < NET_K; k++) _acc[k] += en[k] - eo_[k];
        }
        slot = new_pi;
    }

    // ── 11-cell line patterns (+ cached line codes) ──
    for (int d = 0; d < 3; d++)
        for (int m = -LP_CENTER; m <= LP_CENTER; m++) {
            int xq = qi + m * DIR_Q[d], xr = ri + m * DIR_R[d];
            int& s = _lp[d][xq][xr];
            s += cell_val * POW3_11[LP_CENTER - m];
            _lc[d][xq][xr] = LINE_CODEBOOK[s];
        }

    // ── Candidates ──
    Coord cell = pack(q, r);
    _cand_set.erase(cell);
    _rc_stack.push_back(_cand_rc[qi][ri]);
    _cand_rc[qi][ri] = 0;

    for (const auto& nb : g_nb_offsets) {
        int nq = q + nb.dq, nr = r + nb.dr;
        int nqi = nq + OFF, nri = nr + OFF;
        _cand_rc[nqi][nri]++;
        if (_board[nqi][nri] == 0)
            _cand_set.insert(pack(nq, nr));
    }

    // Place stone
    _board[qi][ri] = player;
    _board_cells.push_back(cell);
    _move_count++;

    if (won) {
        _winner    = player;
        _game_over = true;
    } else {
        _moves_left--;
        if (_moves_left <= 0) {
            _cur_player = (player == P_A) ? P_B : P_A;
            _moves_left = 2;
        }
    }

    // ── Conjunction features: apply class diffs (post-state) ──
    if (_need_acc2) _trunk_cells(ccq, ccr, ncc, +1.f);
    if (!_use_trunk) _conj_diff_apply(ccq, ccr, ncc, ccls);
}

inline void MinimaxBot::_undo(int q, int r, const SavedState& st, int8_t player) {
    int qi = q + OFF, ri = r + OFF;

    // ── Conjunction features: snapshot affected cell classes (pre-undo) ──
    int ccq[32], ccr[32], ccls[32];
    int ncc = _collect_conj_cells(qi, ri, ccq, ccr);
    if (_need_acc2) _trunk_cells(ccq, ccr, ncc, -1.f);
    if (!_use_trunk) _conj_snapshot(ccq, ccr, ncc, ccls);

    // Remove stone
    _board[qi][ri] = 0;
    _board_cells.pop_back();
    _move_count--;
    _cur_player = st.cur_player;
    _moves_left = st.moves_left;
    _winner     = st.winner;
    _game_over  = st.game_over;

    // Zobrist
    _hash ^= get_zobrist(q, r, player);

    int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

    // ── 6-cell windows ──
    if (player == P_A) {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.first--;
            if (counts.first < 4) _hot_a.erase(wo.d_idx, sqi, sri);
        }
    } else {
        for (const auto& wo : g_win_offsets) {
            int sqi = qi - wo.oq, sri = ri - wo.or_;
            auto& counts = _wc[wo.d_idx][sqi][sri];
            counts.second--;
            if (counts.second < 4) _hot_b.erase(wo.d_idx, sqi, sri);
        }
    }

    // ── N-cell eval windows ──
    const double* pv = _pv.data();
    for (const auto& eo : _eval_offsets) {
        int sqi = qi - eo.oq, sri = ri - eo.or_;
        int& slot = _wp[eo.d_idx][sqi][sri];
        int old_pi = slot;
        int new_pi = old_pi - cell_val * _pow3[eo.k];
        _eval_score += pv[new_pi] - pv[old_pi];
        if (_need_acc2) {
            const float* en = TRK_EW[new_pi];
            const float* eo_ = TRK_EW[old_pi];
            for (int k = 0; k < TRK_K; k++) _acc2[k] += en[k] - eo_[k];
        }
        if (!_use_trunk) {
            const float* en = NET_EW[new_pi];
            const float* eo_ = NET_EW[old_pi];
            for (int k = 0; k < NET_K; k++) _acc[k] += en[k] - eo_[k];
        }
        slot = new_pi;
    }

    // ── 11-cell line patterns (+ cached line codes) ──
    for (int d = 0; d < 3; d++)
        for (int m = -LP_CENTER; m <= LP_CENTER; m++) {
            int xq = qi + m * DIR_Q[d], xr = ri + m * DIR_R[d];
            int& s = _lp[d][xq][xr];
            s -= cell_val * POW3_11[LP_CENTER - m];
            _lc[d][xq][xr] = LINE_CODEBOOK[s];
        }

    // ── Candidates ──
    for (const auto& nb : g_nb_offsets) {
        int nq = q + nb.dq, nr = r + nb.dr;
        int nqi = nq + OFF, nri = nr + OFF;
        _cand_rc[nqi][nri]--;
        if (_cand_rc[nqi][nri] == 0)
            _cand_set.erase(pack(nq, nr));
    }
    int saved_rc = _rc_stack.back();
    _rc_stack.pop_back();
    if (saved_rc > 0) {
        Coord cell = pack(q, r);
        _cand_rc[qi][ri] = saved_rc;
        _cand_set.insert(cell);
    }

    // ── Conjunction features: apply class diffs (post-undo) ──
    if (_need_acc2) _trunk_cells(ccq, ccr, ncc, +1.f);
    if (!_use_trunk) _conj_diff_apply(ccq, ccr, ncc, ccls);
}

// ────────────────────────────────────────────────────────────────
//  Turn make / undo
// ────────────────────────────────────────────────────────────────
inline int MinimaxBot::_make_turn(const Turn& turn, UndoStep steps[2]) {
    int q1 = pack_q(turn.first),  r1 = pack_r(turn.first);
    int q2 = pack_q(turn.second), r2 = pack_r(turn.second);

    steps[0] = {turn.first, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
    _make(q1, r1);
    if (_game_over) return 1;

    steps[1] = {turn.second, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
    _make(q2, r2);
    return 2;
}

inline void MinimaxBot::_undo_turn(const UndoStep steps[], int n) {
    for (int i = n - 1; i >= 0; i--)
        _undo(pack_q(steps[i].cell), pack_r(steps[i].cell),
              steps[i].state, steps[i].player);
}

// ────────────────────────────────────────────────────────────────
//  Position loading + eval-array initialisation (shared by
//  get_move and the static_eval / debug_features hooks)
// ────────────────────────────────────────────────────────────────
inline void MinimaxBot::_load_position(const GameState& gs) {
    std::memset(_board, 0, sizeof(_board));
    _board_cells.clear();
    for (const auto& cell : gs.cells) {
        _board[cell.q + OFF][cell.r + OFF] = cell.player;
        _board_cells.push_back(pack(cell.q, cell.r));
    }
    _player     = gs.cur_player;
    _cur_player = gs.cur_player;
    _moves_left = gs.moves_left;
    _move_count = gs.move_count;
    if (_player == P_A) { _cell_a = 1; _cell_b = 2; }
    else                { _cell_a = 2; _cell_b = 1; }
}

inline void MinimaxBot::_init_eval_arrays() {
    std::memset(_wp, 0, sizeof(_wp));
    std::memset(_lp, 0, sizeof(_lp));
    std::memset(_acc, 0, sizeof(_acc));
    std::memset(_acc2, 0, sizeof(_acc2));
    _eval_score = 0.0;

    const double* pv = _pv.data();
    for (Coord c : _board_cells) {
        int bq = pack_q(c), br = pack_r(c);
        int bqi = bq + OFF, bri = br + OFF;
        for (const auto& eo : _eval_offsets) {
            int sqi = bqi - eo.oq, sri = bri - eo.or_;
            int& slot = _wp[eo.d_idx][sqi][sri];
            if (slot != 0) continue;
            int sq = bq - eo.oq, sr = br - eo.or_;
            int d = eo.d_idx;
            int pi = 0;
            bool has = false;
            for (int j = 0; j < _eval_length; j++) {
                int8_t v = _board[sq + j * DIR_Q[d] + OFF][sr + j * DIR_R[d] + OFF];
                if (v != 0) {
                    pi += ((v == P_A) ? _cell_a : _cell_b) * _pow3[j];
                    has = true;
                }
            }
            if (has) {
                slot = pi;
                _eval_score += pv[pi];
                if (_use_trunk) {
                    const float* e = TRK_EW[pi];
                    for (int k = 0; k < TRK_K; k++) _acc2[k] += e[k];
                } else {
                    const float* e = NET_EW[pi];
                    for (int k = 0; k < NET_K; k++) _acc[k] += e[k];
                }
            }
        }
    }

    // 11-cell line patterns + cached line codes
    std::memset(_lc, LINE_CODEBOOK[0], sizeof(_lc));
    for (Coord c : _board_cells) {
        int bqi = pack_q(c) + OFF, bri = pack_r(c) + OFF;
        int8_t v = _board[bqi][bri];
        int cell_val = (v == P_A) ? _cell_a : _cell_b;
        for (int d = 0; d < 3; d++)
            for (int m = -LP_CENTER; m <= LP_CENTER; m++) {
                int xq = bqi + m * DIR_Q[d], xr = bri + m * DIR_R[d];
                int& s = _lp[d][xq][xr];
                s += cell_val * POW3_11[LP_CENTER - m];
                _lc[d][xq][xr] = LINE_CODEBOOK[s];
            }
    }

    // Conjunction classes (legacy) / clamped cell activations (trunk)
    // over the bounding box of influence
    if (!_board_cells.empty()) {
        int q0 = ARR, q1 = -1, r0 = ARR, r1 = -1;
        for (Coord c : _board_cells) {
            int bqi = pack_q(c) + OFF, bri = pack_r(c) + OFF;
            q0 = std::min(q0, bqi); q1 = std::max(q1, bqi);
            r0 = std::min(r0, bri); r1 = std::max(r1, bri);
        }
        q0 = std::max(q0 - LP_CENTER, 0); q1 = std::min(q1 + LP_CENTER, ARR - 1);
        r0 = std::max(r0 - LP_CENTER, 0); r1 = std::min(r1 + LP_CENTER, ARR - 1);
        for (int qi = q0; qi <= q1; qi++)
            for (int ri = r0; ri <= r1; ri++) {
                if (_need_acc2) _trunk_cell(qi, ri, +1.f);
                if (_use_trunk) continue;
                int cls = _conj_class(qi, ri);
                if (cls < 0) continue;
                const float* e = NET_EC[cls];
                for (int k = 0; k < NET_K; k++) _acc[k] += e[k];
            }
    }
}

inline double MinimaxBot::static_eval(const GameState& gs) {
    _load_position(gs);
    _init_eval_arrays();
    return _leaf_eval();
}

inline std::pair<std::vector<std::pair<int,int>>, std::vector<std::pair<int,int>>>
MinimaxBot::debug_features(const GameState& gs) {
    _load_position(gs);
    _init_eval_arrays();

    flat_map<int, int> wf, cf;
    for (int d = 0; d < 3; d++)
        for (int qi = 0; qi < ARR; qi++)
            for (int ri = 0; ri < ARR; ri++) {
                int pi = _wp[d][qi][ri];
                if (pi != 0) wf[pi]++;
            }
    for (int qi = 0; qi < ARR; qi++)
        for (int ri = 0; ri < ARR; ri++) {
            int cls = _conj_class(qi, ri);
            if (cls >= 0) cf[cls]++;
        }

    std::vector<std::pair<int,int>> wv(wf.begin(), wf.end());
    std::vector<std::pair<int,int>> cv(cf.begin(), cf.end());
    return {wv, cv};
}

// ────────────────────────────────────────────────────────────────
//  Move delta
// ────────────────────────────────────────────────────────────────
inline double MinimaxBot::_move_delta(int q, int r, bool is_a) const {
    int8_t cell_val = is_a ? _cell_a : _cell_b;
    const double* pv = _pv.data();
    int qi = q + OFF, ri = r + OFF;
    double delta = 0.0;
    for (const auto& eo : _eval_offsets) {
        int old_pi = _wp[eo.d_idx][qi - eo.oq][ri - eo.or_];
        int new_pi = old_pi + cell_val * _pow3[eo.k];
        delta += pv[new_pi] - pv[old_pi];
    }
    return delta;
}

} // namespace opt
