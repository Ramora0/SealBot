/*
 * search.h -- MinimaxBot search: get_move, extract_pv, minimax, quiescence.
 */
#pragma once

#include "bot.h"

namespace opt {

// ────────────────────────────────────────────────────────────────
//  Main entry point
// ────────────────────────────────────────────────────────────────
// Position loading shared by get_move and lazy-SMP helpers: board arrays,
// zobrist, windows, eval accumulators, candidates.
inline void MinimaxBot::_setup_position(const GameState& gs) {
    // ── Clear arrays ──
    std::memset(_board, 0, sizeof(_board));
    std::memset(_wc, 0, sizeof(_wc));
    std::memset(_wp, 0, sizeof(_wp));
    std::memset(_cand_rc, 0, sizeof(_cand_rc));
    _board_cells.clear();
    _hot_a.clear();
    _hot_b.clear();
    _cand_set.clear();
    _rc_stack.clear();

    // ── Populate board from GameState ──
    for (const auto& cell : gs.cells) {
        _board[cell.q + OFF][cell.r + OFF] = cell.player;
        _board_cells.push_back(pack(cell.q, cell.r));
    }

    _cur_player = gs.cur_player;
    _moves_left = gs.moves_left;
    _move_count = gs.move_count;
    _winner     = P_NONE;
    _game_over  = false;

    // ── Player tracking ──
    if (_cur_player != _player) {
        _history.clear();
        std::memset(_killers, 0, sizeof(_killers));
    }
    _player    = _cur_player;
    _nodes     = 0;
    _ply       = 0;
    last_depth = 0;
    last_score = 0;
    last_ebf   = 0;

    // ── Zobrist ──
    _hash = 0;
    for (Coord c : _board_cells)
        _hash ^= get_zobrist(pack_q(c), pack_r(c),
                              _board[pack_q(c) + OFF][pack_r(c) + OFF]);

    // ── Cell value mapping ──
    if (_player == P_A) { _cell_a = 1; _cell_b = 2; }
    else                { _cell_a = 2; _cell_b = 1; }

    // ── Init 6-cell windows ──
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

    // ── Init N-cell eval windows + NNUE accumulator (wp, lp, acc) ──
    _init_eval_arrays();

    // ── Init candidates ──
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

// Lazy-SMP helper: same position, own killers/history/board state, SHARED
// transposition table. Runs its own iterative deepening until deadline or
// the main thread's stop flag; its value is the TT entries it leaves.
inline void MinimaxBot::_helper_loop(GameState gs, Clock::time_point deadline,
                                     std::atomic<bool>* stop, int offset) {
    _deadline = deadline;
    _stop_ext = stop;
    _setup_position(gs);
    if (_cand_set.empty()) { _stop_ext = nullptr; return; }
    bool maximizing = (_cur_player == _player);
    auto turns = _generate_turns();
    if (turns.empty()) { _stop_ext = nullptr; return; }
    // Stagger: odd helpers start one ply deeper to diversify the shared TT.
    for (int depth = 1 + (offset & 1); depth <= max_depth; depth++) {
        try {
            auto rr = _search_root(turns, depth);
            auto& scores = rr.second;
            std::sort(turns.begin(), turns.end(),
                [&scores, maximizing](const Turn& a, const Turn& b) {
                    double sa = 0, sb = 0;
                    auto ia = scores.find(a); if (ia != scores.end()) sa = ia->second;
                    auto ib = scores.find(b); if (ib != scores.end()) sb = ib->second;
                    return maximizing ? (sa > sb) : (sa < sb);
                });
            auto si = scores.find(rr.first);
            if (si != scores.end() && std::abs(si->second) >= WIN_THRESHOLD)
                break;
        } catch (const TimeUp&) {
            break;
        }
    }
    _stop_ext = nullptr;
}

inline MoveResult MinimaxBot::get_move(const GameState& gs) {
    if (gs.cells.empty())
        return {0, 0, 0, 0, 1};

    _setup_position(gs);

    // ── Deadline (bit 8 reserves a slice for the post-search veto) ──
    double tl_search = (vcf_mode & 8) ? time_limit * 0.82 : time_limit;
    _deadline = Clock::now() + std::chrono::microseconds(
                    static_cast<int64_t>(tl_search * 1000000.0));

    if (_cand_set.empty())
        return {0, 0, 0, 0, 1};

    bool maximizing = (_cur_player == _player);
    auto turns = _generate_turns();
    if (turns.empty())
        return {0, 0, 0, 0, 1};

    // ── VCF root probes ──
    if (vcf_mode & 1) {
        // Attack: play a proven forced win immediately.
        Turn wt{};
        if (forced_win(_cur_player, _moves_left, vcf_k, &wt) == 1) {
            last_depth = 99;
            last_score = (WIN_SCORE - 1);
            return {pack_q(wt.first),  pack_r(wt.first),
                    pack_q(wt.second), pack_r(wt.second), 2};
        }
    }
    if ((vcf_mode & 2) && turns.size() > 3) {
        // Defense: drop turns after which the opponent has a proven
        // forced win (bounded probe per turn). Keep at least 3 turns.
        int budget_save = vcf_node_budget;
        vcf_node_budget = std::min(8000, std::max(800, budget_save / 6));
        // Respect the move clock: stop filtering once 40% of the budget is
        // spent (unfiltered turns pass through; the search handles them).
        auto vcf_cutoff = Clock::now() +
            std::chrono::microseconds(static_cast<int64_t>(
                time_limit * 400000.0));
        std::vector<Turn> safe;
        safe.reserve(turns.size());
        for (const auto& t : turns) {
            if (Clock::now() >= vcf_cutoff) { safe.push_back(t); continue; }
            UndoStep steps[2];
            int n = _make_turn(t, steps);
            bool losing = false;
            if (!_game_over)
                losing = (forced_win(_cur_player, _moves_left,
                                     std::max(4, vcf_k / 2),
                                     nullptr) == 1);
            _undo_turn(steps, n);
            if (!losing) safe.push_back(t);
        }
        vcf_node_budget = budget_save;
        // All-losing => keep the originals (search picks longest resistance).
        if (!safe.empty())
            turns = std::move(safe);
    }

    Turn best_move = turns[0];

    // ── Lazy SMP: launch helper searchers sharing the TT ──
    std::vector<std::thread> smp_pool;
    std::atomic<bool> smp_stop{false};
    if (threads > 1) {
        while (_helpers.size() < static_cast<size_t>(threads - 1))
            _helpers.push_back(std::make_unique<MinimaxBot>(time_limit));
        for (int i = 0; i < threads - 1; i++) {
            MinimaxBot* h = _helpers[i].get();
            h->_clone_config_from(*this);
            smp_pool.emplace_back(&MinimaxBot::_helper_loop, h, gs,
                                  _deadline, &smp_stop, i);
        }
    }

    // ── Save state for TimeUp rollback ──
    if (!_saved) _saved = std::make_unique<SavedArrays>();
    std::memcpy(_saved->board, _board, sizeof(_board));
    std::memcpy(_saved->wc, _wc, sizeof(_wc));
    std::memcpy(_saved->wp, _wp, sizeof(_wp));
    std::memcpy(_saved->acc, _acc, sizeof(_acc));
    std::memcpy(_saved->acc2, _acc2, sizeof(_acc2));
    std::memcpy(_saved->lp, _lp, sizeof(_lp));
    std::memcpy(_saved->lc, _lc, sizeof(_lc));
    std::memcpy(_saved->cand_rc, _cand_rc, sizeof(_cand_rc));
    std::memcpy(_saved->cand_bits, _cand_set.bits, sizeof(_cand_set.bits));
    _saved->cand_vec = _cand_set.vec;
    std::memcpy(_saved->hot_a_bits, _hot_a.bits, sizeof(_hot_a.bits));
    _saved->hot_a_vec = _hot_a.vec;
    std::memcpy(_saved->hot_b_bits, _hot_b.bits, sizeof(_hot_b.bits));
    _saved->hot_b_vec = _hot_b.vec;
    _saved->board_cells = _board_cells;
    auto saved_st       = SavedState{_cur_player, _moves_left, _winner, _game_over};
    int  saved_mc       = _move_count;
    uint64_t saved_hash = _hash;
    double   saved_eval = _eval_score;

    for (int depth = 1; depth <= max_depth; depth++) {
        try {
            int nb4 = _nodes;
            auto root_result = _search_root(turns, depth);
            Turn result = root_result.first;
            auto& scores = root_result.second;
            best_move  = result;
            last_depth = depth;
            auto si = scores.find(result);
            last_score = (si != scores.end()) ? si->second : 0.0;
            int nthis = _nodes - nb4;
            if (nthis > 1)
                last_ebf = std::round(std::pow(static_cast<double>(nthis),
                                               1.0 / depth) * 10.0) / 10.0;
            std::sort(turns.begin(), turns.end(),
                [&scores, maximizing](const Turn& a, const Turn& b) {
                    double sa = 0, sb = 0;
                    auto ia = scores.find(a); if (ia != scores.end()) sa = ia->second;
                    auto ib = scores.find(b); if (ib != scores.end()) sb = ib->second;
                    return maximizing ? (sa > sb) : (sa < sb);
                });
            if (std::abs(last_score) >= WIN_THRESHOLD) break;
        } catch (const TimeUp&) {
            // Harvest the aborted iteration: its best fully-searched move
            // supersedes the previous iteration's choice (the previous best
            // is always searched first, so this is monotone information).
            if (_root_partial_valid) {
                best_move  = _root_partial_best;
                last_score = _root_partial_score;
                _root_partial_valid = false;
            }
            std::memcpy(_board, _saved->board, sizeof(_board));
            std::memcpy(_wc, _saved->wc, sizeof(_wc));
            std::memcpy(_wp, _saved->wp, sizeof(_wp));
            std::memcpy(_acc, _saved->acc, sizeof(_acc));
            std::memcpy(_acc2, _saved->acc2, sizeof(_acc2));
            std::memcpy(_lp, _saved->lp, sizeof(_lp));
            std::memcpy(_lc, _saved->lc, sizeof(_lc));
            std::memcpy(_cand_rc, _saved->cand_rc, sizeof(_cand_rc));
            std::memcpy(_cand_set.bits, _saved->cand_bits, sizeof(_cand_set.bits));
            _cand_set.vec = std::move(_saved->cand_vec);
            std::memcpy(_hot_a.bits, _saved->hot_a_bits, sizeof(_hot_a.bits));
            _hot_a.vec = std::move(_saved->hot_a_vec);
            std::memcpy(_hot_b.bits, _saved->hot_b_bits, sizeof(_hot_b.bits));
            _hot_b.vec = std::move(_saved->hot_b_vec);
            _board_cells = std::move(_saved->board_cells);
            _move_count = saved_mc;
            _cur_player = saved_st.cur_player;
            _moves_left = saved_st.moves_left;
            _winner     = saved_st.winner;
            _game_over  = saved_st.game_over;
            _hash       = saved_hash;
            _eval_score = saved_eval;
            break;
        }
    }

    smp_stop.store(true);
    for (auto& t : smp_pool) t.join();

    // ── Post-search deep veto (bit 8): the analyzer showed 100% of losses
    // are deep forced wins we walked into. Prove the CHOSEN move doesn't
    // hand strix a forced win; if it does, walk down the root ordering.
    if ((vcf_mode & 8) && turns.size() > 1) {
        auto veto_deadline = Clock::now() + std::chrono::microseconds(
            static_cast<int64_t>(time_limit * 0.16 * 1000000.0));
        std::vector<Turn> order;
        order.push_back(best_move);
        for (const auto& t : turns)
            if (!(t == best_move)) order.push_back(t);
        int probes = 0;
        for (const auto& t : order) {
            if (probes >= 5 || Clock::now() >= veto_deadline) break;
            UndoStep steps[2];
            int n = _make_turn(t, steps);
            bool losing = false;
            if (!_game_over) {
                probes++;
                // Tiered: the search's chosen move gets one DEEP probe
                // (post-mortems prove our loss entries at k<=16 while
                // shallow probes miss them); alternatives get k-2.
                int kk = (probes == 1) ? vcf_k + 3
                                       : std::max(6, vcf_k - 2);
                int bsave = vcf_node_budget;
                if (probes == 1) vcf_node_budget = bsave * 2;
                losing = (forced_win(_cur_player, _moves_left, kk,
                                     nullptr) == 1);
                vcf_node_budget = bsave;
            }
            _undo_turn(steps, n);
            if (!losing) { best_move = t; break; }
        }
    }

    return {pack_q(best_move.first),  pack_r(best_move.first),
            pack_q(best_move.second), pack_r(best_move.second), 2};
}

// ────────────────────────────────────────────────────────────────
//  PV extraction from TT after search
// ────────────────────────────────────────────────────────────────
inline std::vector<PVStep> MinimaxBot::extract_pv() {
    std::vector<PVStep> pv;
    std::vector<std::pair<UndoStep[2], int>> undo_stack;
    flat_set<uint64_t> seen;
    _ply = 0;

    while (!_game_over) {
        uint64_t ttk = _tt_key();
        if (seen.count(ttk)) break;
        seen.insert(ttk);

        int8_t player = _cur_player;

        // 1. Check instant win for current player
        auto [found, wt] = _find_instant_win(player);
        if (found) {
            undo_stack.push_back({});
            auto& back = undo_stack.back();
            back.second = _make_turn(wt, back.first);
            _ply++;
            PVStep step;
            step.player = player;
            for (int i = 0; i < back.second; i++) {
                Coord c = back.first[i].cell;
                step.moves.push_back({pack_q(c), pack_r(c)});
            }
            pv.push_back(std::move(step));
            break;
        }

        // 2. TT entry with a best move
        TTView tv{};
        bool tth = _tt_probe(ttk, tv);
        if (tth && tv.has_move) {
            double sc = _tt_adjust_load(tv.score);
            if (std::abs(sc) < WIN_THRESHOLD) break;
            Turn best_turn = tv.move;
            undo_stack.push_back({});
            auto& back = undo_stack.back();
            back.second = _make_turn(best_turn, back.first);
            _ply++;
            PVStep step;
            step.player = player;
            for (int i = 0; i < back.second; i++) {
                Coord c = back.first[i].cell;
                step.moves.push_back({pack_q(c), pack_r(c)});
            }
            pv.push_back(std::move(step));
            if (_game_over) break;
            continue;
        }

        // 3. No TT move — try threat-based defense (for defender nodes)
        int8_t opponent = (player == P_A) ? P_B : P_A;
        auto opp_threats = _find_threat_cells(opponent);
        auto my_threats  = _find_threat_cells(player);
        auto threat_turns = _generate_threat_turns(my_threats, opp_threats);
        if (threat_turns.empty()) break;

        // Pick response that maximizes opponent's work (prefer longest survival)
        Turn best_response = threat_turns[0];
        bool found_resp = false;
        double best_surv = -INF_SCORE;
        bool maximizing = (player == _player);
        for (const auto& turn : threat_turns) {
            UndoStep tmp[2];
            int n = _make_turn(turn, tmp);
            _ply++;
            if (_game_over) {
                _ply--;
                _undo_turn(tmp, n);
                continue;
            }
            // Check TT for score after this response
            TTView tv2{};
            double surv;
            if (_tt_probe(_tt_key(), tv2)) {
                surv = _tt_adjust_load(tv2.score);
            } else {
                // No TT — use instant win check as proxy
                auto [ofw, _owt] = _find_instant_win(opponent);
                surv = ofw ? (maximizing ? -WIN_SCORE + _ply : WIN_SCORE - _ply) : 0.0;
            }
            _ply--;
            _undo_turn(tmp, n);
            // Defender wants to minimize (if opponent is maximizer) or maximize
            bool better = maximizing ? (surv > best_surv) : (surv < best_surv);
            if (!found_resp || better) {
                best_response = turn;
                best_surv = surv;
                found_resp = true;
            }
        }
        if (!found_resp) break;

        undo_stack.push_back({});
        auto& back = undo_stack.back();
        back.second = _make_turn(best_response, back.first);
        _ply++;
        PVStep step;
        step.player = player;
        for (int i = 0; i < back.second; i++) {
            Coord c = back.first[i].cell;
            step.moves.push_back({pack_q(c), pack_r(c)});
        }
        pv.push_back(std::move(step));
        if (_game_over) break;
    }

    // Undo everything
    for (auto it2 = undo_stack.rbegin(); it2 != undo_stack.rend(); ++it2) {
        _ply--;
        _undo_turn(it2->first, it2->second);
    }

    return pv;
}

// ────────────────────────────────────────────────────────────────
//  Quiescence search
// ────────────────────────────────────────────────────────────────
inline double MinimaxBot::_quiescence(double alpha, double beta, int qdepth) {
    _check_time();

    if (_game_over) {
        if (_winner == _player)    return  WIN_SCORE - _ply;
        if (_winner != P_NONE)     return -WIN_SCORE + _ply;
        return 0.0;
    }

    auto [found, wt] = _find_instant_win(_cur_player);
    if (found) {
        UndoStep steps[2];
        int n = _make_turn(wt, steps);
        _ply++;
        double sc = (_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply);
        _ply--;
        _undo_turn(steps, n);
        return sc;
    }

    double stand_pat = _leaf_eval();
    int8_t current  = _cur_player;
    int8_t opponent = (current == P_A) ? P_B : P_A;

    auto my_threats  = _find_threat_cells(current);
    auto opp_threats = _find_threat_cells(opponent);

    if ((my_threats.empty() && opp_threats.empty()) || qdepth <= 0)
        return stand_pat;

    bool maximizing = (current == _player);
    if (maximizing) {
        if (stand_pat >= beta) return stand_pat;
        alpha = std::max(alpha, stand_pat);
    } else {
        if (stand_pat <= alpha) return stand_pat;
        beta = std::min(beta, stand_pat);
    }

    auto threat_turns = _generate_threat_turns(my_threats, opp_threats);
    if (threat_turns.empty()) return stand_pat;

    double value = stand_pat;
    if (maximizing) {
        for (const auto& turn : threat_turns) {
            UndoStep steps[2];
            int nm = _make_turn(turn, steps);
            _ply++;
            double cv = _game_over
                ? ((_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply))
                : _quiescence(alpha, beta, qdepth - 1);
            _ply--;
            _undo_turn(steps, nm);
            if (cv > value) value = cv;
            alpha = std::max(alpha, value);
            if (alpha >= beta) break;
        }
    } else {
        for (const auto& turn : threat_turns) {
            UndoStep steps[2];
            int nm = _make_turn(turn, steps);
            _ply++;
            double cv = _game_over
                ? ((_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply))
                : _quiescence(alpha, beta, qdepth - 1);
            _ply--;
            _undo_turn(steps, nm);
            if (cv < value) value = cv;
            beta = std::min(beta, value);
            if (alpha >= beta) break;
        }
    }
    return value;
}

// ────────────────────────────────────────────────────────────────
//  Root search
// ────────────────────────────────────────────────────────────────
inline std::pair<Turn, flat_map<Turn, double, TurnHash>>
MinimaxBot::_search_root(std::vector<Turn>& turns, int depth) {
    bool maximizing = (_cur_player == _player);
    Turn best = turns[0];
    double alpha = -INF_SCORE, beta = INF_SCORE;

    flat_map<Turn, double, TurnHash> scores;
    scores.reserve(turns.size());

    _root_partial_valid = false;
    for (const auto& turn : turns) {
        _check_time();
        UndoStep steps[2];
        int n = _make_turn(turn, steps);
        _ply++;
        double sc;
        if (_game_over)
            sc = (_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply);
        else
            sc = _minimax(depth - 1, alpha, beta);
        _ply--;
        _undo_turn(steps, n);
        scores[turn] = sc;

        if (maximizing && sc > alpha)  { alpha = sc; best = turn; }
        if (!maximizing && sc < beta)  { beta  = sc; best = turn; }
        // Partial-iteration harvest: remember the best fully-searched move
        // of THIS iteration so a TimeUp mid-iteration doesn't discard it.
        _root_partial_best  = best;
        _root_partial_score = maximizing ? alpha : beta;
        _root_partial_valid = true;
    }

    double best_sc = maximizing ? alpha : beta;
    _tt_store_entry(_tt_key(), depth, _tt_adjust_store(best_sc), TT_EXACT, best, true);
    return {best, std::move(scores)};
}

// ────────────────────────────────────────────────────────────────
//  Minimax
// ────────────────────────────────────────────────────────────────
inline double MinimaxBot::_minimax(int depth, double alpha, double beta) {
    _check_time();

    if (_game_over) {
        if (_winner == _player)    return  WIN_SCORE - _ply;
        if (_winner != P_NONE)     return -WIN_SCORE + _ply;
        return 0.0;
    }

    uint64_t ttk = _tt_key();
    Turn tt_move{};
    bool has_tt_move = false;

    TTView tte{};
    if (_tt_probe(ttk, tte)) {
        has_tt_move = tte.has_move;
        tt_move     = tte.move;
        if (tte.depth >= depth) {
            double sc = _tt_adjust_load(tte.score);
            if (tte.flag == TT_EXACT) return sc;
            if (tte.flag == TT_LOWER) alpha = std::max(alpha, sc);
            if (tte.flag == TT_UPPER) beta  = std::min(beta,  sc);
            if (alpha >= beta) return sc;
        }
    }

    if (depth == 0) {
        double sc = _quiescence(alpha, beta, MAX_QDEPTH);
        int8_t qflag;
        if (sc <= alpha) qflag = TT_UPPER;
        else if (sc >= beta) qflag = TT_LOWER;
        else qflag = TT_EXACT;
        _tt_store_entry(ttk, 0, _tt_adjust_store(sc), qflag, Turn{}, false);
        return sc;
    }

    // Instant win for current player
    {
        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) {
            UndoStep steps[2];
            int n = _make_turn(wt, steps);
            _ply++;
            double sc = (_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply);
            _ply--;
            _undo_turn(steps, n);
            _tt_store_entry(ttk, depth, _tt_adjust_store(sc), TT_EXACT, wt, true);
            return sc;
        }
    }

    // VCF interior probe: proven forced win for the mover terminates the
    // node without expansion (tactical depth grafted onto shallow search).
    if ((vcf_mode & 4) && depth >= 2) {
        int budget_save = vcf_node_budget;
        vcf_node_budget = std::max(400, budget_save / 12);
        Turn vt{};
        int fw = forced_win(_cur_player, _moves_left, 3, &vt);
        vcf_node_budget = budget_save;
        if (fw == 1) {
            double sc = (_cur_player == _player) ? (WIN_SCORE - _ply - 4)
                                                 : (-WIN_SCORE + _ply + 4);
            _tt_store_entry(ttk, depth, _tt_adjust_store(sc), TT_EXACT, vt,
                            true);
            return sc;
        }
    }

    // Opponent instant win -> check if blockable
    int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
    {
        auto [opp_found, opp_wt] = _find_instant_win(opponent);
        if (opp_found) {
            int p_idx = (opponent == P_A) ? 0 : 1;
            const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;
            std::vector<flat_set<Coord>> must_hit;
            for (const auto& he : hot.vec) {
                auto& counts = _wc[he.d][he.qi][he.ri];
                int mc = (p_idx == 0) ? counts.first  : counts.second;
                int oc = (p_idx == 0) ? counts.second : counts.first;
                if (mc < WIN_LENGTH - 2 || oc != 0) continue;

                int sq = he.qi - OFF, sr = he.ri - OFF;
                int dq = DIR_Q[he.d], dr = DIR_R[he.d];
                flat_set<Coord> empties;
                for (int j = 0; j < WIN_LENGTH; j++) {
                    int cq = sq + j * dq, cr = sr + j * dr;
                    if (_board[cq + OFF][cr + OFF] == 0)
                        empties.insert(pack(cq, cr));
                }
                must_hit.push_back(std::move(empties));
            }
            if (must_hit.size() > 1) {
                flat_set<Coord> all_cells;
                for (const auto& s : must_hit) all_cells.insert(s.begin(), s.end());
                bool can_block = false;
                for (Coord c1 : all_cells) {
                    for (Coord c2 : all_cells) {
                        bool ok = true;
                        for (const auto& w : must_hit)
                            if (!w.count(c1) && !w.count(c2)) { ok = false; break; }
                        if (ok) { can_block = true; break; }
                    }
                    if (can_block) break;
                }
                if (!can_block) {
                    // Opponent has unblockable win — they win next turn
                    double sc = (opponent != _player)
                        ? (-WIN_SCORE + _ply + 1) : (WIN_SCORE - _ply - 1);
                    _tt_store_entry(ttk, depth, _tt_adjust_store(sc), TT_EXACT, Turn{}, false);
                    return sc;
                }
            }
        }
    }

    double orig_alpha = alpha, orig_beta = beta;
    bool maximizing = (_cur_player == _player);

    // Generate candidates and turns
    std::vector<Turn> turns;
    {
        std::vector<Coord> cands(_cand_set.begin(), _cand_set.end());
        if (cands.size() < 2) {
            if (cands.empty()) {
                double sc = _leaf_eval();
                _tt_store_entry(ttk, depth, sc, TT_EXACT, Turn{}, false);
                return sc;
            }
            turns = {{cands[0], cands[0]}};
        } else {
            // Strix-distilled policy selection + linear-delta safety net —
            // deterministic and independent of history heuristic. Bit 0
            // governs our-side interior selection, bit 3 the opponent's.
            bool is_a = (_cur_player == P_A);
            int cap = no_cand_cap ? static_cast<int>(cands.size()) : cand_cap;
            bool use_pol = maximizing ? (policy_mode & 1) != 0
                                      : (policy_mode & 8) != 0;
            // bit 4: at opponent (min) nodes, widen the delta selection
            // with top policy extras (delta_keep) — union, never replace.
            int saved_keep = delta_keep;
            if (!maximizing && !use_pol && !(policy_mode & 16))
                delta_keep = 0;
            if (maximizing && !use_pol)
                delta_keep = 0;
            _select_candidates(cands, cap, maximizing, is_a, use_pol);
            delta_keep = saved_keep;

            int n = static_cast<int>(cands.size());
            if (no_cand_cap) {
                turns.reserve(n * (n - 1) / 2);
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                        turns.push_back({cands[i], cands[j]});
            } else {
                turns.reserve(g_inner_pairs.size());
                for (const auto& [pi, pj] : g_inner_pairs) {
                    if (pj >= n) continue;
                    turns.push_back({cands[pi], cands[pj]});
                }
            }
            turns = _filter_turns_by_threats(turns);
        }
    }

    if (turns.empty()) {
        double sc = _leaf_eval();
        _tt_store_entry(ttk, depth, sc, TT_EXACT, Turn{}, false);
        return sc;
    }

    // TT move ordering
    if (has_tt_move) {
        for (size_t i = 0; i < turns.size(); i++)
            if (turns[i] == tt_move) { std::swap(turns[0], turns[i]); break; }
    }

    // Killer move ordering (after TT move)
    if (_ply < MAX_KILLERS_PLY) {
        size_t next = has_tt_move ? 1 : 0;
        for (int ki = 0; ki < 2; ki++) {
            const Turn& killer = _killers[_ply][ki];
            if (killer.first == 0 && killer.second == 0) continue;
            if (has_tt_move && killer == tt_move) continue;
            for (size_t i = next; i < turns.size(); i++)
                if (turns[i] == killer) {
                    std::swap(turns[next], turns[i]);
                    next++;
                    break;
                }
        }
    }

    Turn best_move{};
    double value;

    if (maximizing) {
        value = -INF_SCORE;
        for (const auto& turn : turns) {
            UndoStep steps[2];
            int n = _make_turn(turn, steps);
            _ply++;
            double cv = _game_over
                ? ((_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply))
                : _minimax(depth - 1, alpha, beta);
            _ply--;
            _undo_turn(steps, n);
            if (cv > value) { value = cv; best_move = turn; }
            alpha = std::max(alpha, value);
            if (alpha >= beta) {
                _history[turn.first]  += depth * depth;
                _history[turn.second] += depth * depth;
                _store_killer(_ply, turn);
                break;
            }
        }
    } else {
        value = INF_SCORE;
        for (const auto& turn : turns) {
            UndoStep steps[2];
            int n = _make_turn(turn, steps);
            _ply++;
            double cv = _game_over
                ? ((_winner == _player) ? (WIN_SCORE - _ply) : (-WIN_SCORE + _ply))
                : _minimax(depth - 1, alpha, beta);
            _ply--;
            _undo_turn(steps, n);
            if (cv < value) { value = cv; best_move = turn; }
            beta = std::min(beta, value);
            if (alpha >= beta) {
                _history[turn.first]  += depth * depth;
                _history[turn.second] += depth * depth;
                _store_killer(_ply, turn);
                break;
            }
        }
    }

    int8_t flag;
    if      (value <= orig_alpha) flag = TT_UPPER;
    else if (value >= orig_beta)  flag = TT_LOWER;
    else                          flag = TT_EXACT;
    _tt_store_entry(ttk, depth, _tt_adjust_store(value), flag, best_move, true);
    return value;
}

} // namespace opt
