/*
 * movegen.h -- MinimaxBot win/threat detection and turn generation.
 */
#pragma once

#include "bot.h"

namespace opt {

// ────────────────────────────────────────────────────────────────
//  Win / threat detection
// ────────────────────────────────────────────────────────────────
inline std::pair<bool, Turn> MinimaxBot::_find_instant_win(int8_t player) const {
    int p_idx = (player == P_A) ? 0 : 1;
    const auto& hot = (player == P_A) ? _hot_a : _hot_b;

    for (const auto& he : hot.vec) {
        auto& counts = _wc[he.d][he.qi][he.ri];
        int my_count  = (p_idx == 0) ? counts.first : counts.second;
        int opp_count = (p_idx == 0) ? counts.second : counts.first;

        if (my_count >= WIN_LENGTH - 2 && opp_count == 0) {
            int sq = he.qi - OFF, sr = he.ri - OFF;
            int dq = DIR_Q[he.d], dr = DIR_R[he.d];

            Coord cells[WIN_LENGTH];
            int n = 0;
            for (int j = 0; j < WIN_LENGTH; j++) {
                int cq = sq + j * dq, cr = sr + j * dr;
                if (_board[cq + OFF][cr + OFF] == 0)
                    cells[n++] = pack(cq, cr);
            }
            if (n == 1) {
                Coord other = cells[0];
                for (Coord c : _cand_set)
                    if (c != cells[0]) { other = c; break; }
                return {true, {coord_min(cells[0], other),
                               coord_max(cells[0], other)}};
            }
            if (n == 2) {
                return {true, {coord_min(cells[0], cells[1]),
                               coord_max(cells[0], cells[1])}};
            }
        }
    }
    return {false, {}};
}

inline flat_set<Coord> MinimaxBot::_find_threat_cells(int8_t player) const {
    flat_set<Coord> threats;
    int p_idx = (player == P_A) ? 0 : 1;
    const auto& hot = (player == P_A) ? _hot_a : _hot_b;

    for (const auto& he : hot.vec) {
        auto& counts = _wc[he.d][he.qi][he.ri];
        int opp_count = (p_idx == 0) ? counts.second : counts.first;
        if (opp_count != 0) continue;

        int sq = he.qi - OFF, sr = he.ri - OFF;
        int dq = DIR_Q[he.d], dr = DIR_R[he.d];

        for (int j = 0; j < WIN_LENGTH; j++) {
            int cq = sq + j * dq, cr = sr + j * dr;
            if (_board[cq + OFF][cr + OFF] == 0)
                threats.insert(pack(cq, cr));
        }
    }
    return threats;
}

inline std::vector<Turn> MinimaxBot::_filter_turns_by_threats(
        const std::vector<Turn>& turns) const {
    int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
    int p_idx = (opponent == P_A) ? 0 : 1;
    const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;

    std::vector<flat_set<Coord>> must_hit;
    for (const auto& he : hot.vec) {
        auto& counts = _wc[he.d][he.qi][he.ri];
        int my_count  = (p_idx == 0) ? counts.first  : counts.second;
        int opp_count = (p_idx == 0) ? counts.second : counts.first;
        if (my_count < WIN_LENGTH - 2 || opp_count != 0) continue;

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
    if (must_hit.empty()) return turns;

    std::vector<Turn> out;
    out.reserve(turns.size());
    for (const auto& t : turns) {
        bool ok = true;
        for (const auto& w : must_hit) {
            if (!w.count(t.first) && !w.count(t.second)) {
                ok = false; break;
            }
        }
        if (ok) out.push_back(t);
    }
    return out.empty() ? turns : out;
}

// ────────────────────────────────────────────────────────────────
//  Turn generation
// ────────────────────────────────────────────────────────────────
inline std::vector<Turn> MinimaxBot::_generate_turns() {
    auto [found, wt] = _find_instant_win(_cur_player);
    if (found) return {wt};

    std::vector<Coord> cands(_cand_set.begin(), _cand_set.end());
    if (cands.size() < 2) {
        if (!cands.empty()) return {{cands[0], cands[0]}};
        return {};
    }

    bool maximizing = (_cur_player == _player);
    bool is_a = (_cur_player == P_A);
    int cap = no_cand_cap ? static_cast<int>(cands.size()) : root_cand_cap;
    _select_candidates(cands, cap, maximizing, is_a, (policy_mode & 2) != 0);

    // Colony candidate
    if (!_board_cells.empty()) {
        int64_t sq = 0, sr = 0;
        for (Coord c : _board_cells) { sq += pack_q(c); sr += pack_r(c); }
        int cq = static_cast<int>(sq / static_cast<int64_t>(_board_cells.size()));
        int cr = static_cast<int>(sr / static_cast<int64_t>(_board_cells.size()));
        int max_r = 0;
        for (Coord c : _board_cells) {
            int d = hex_distance(pack_q(c) - cq, pack_r(c) - cr);
            if (d > max_r) max_r = d;
        }
        int cd = max_r + 3;
        int di = static_cast<int>(_hash % 6);
        int col_q = cq + COLONY_DQ[di] * cd;
        int col_r = cr + COLONY_DR[di] * cd;
        if (std::abs(col_q) < OFF && std::abs(col_r) < OFF &&
            _board[col_q + OFF][col_r + OFF] == 0)
            cands.push_back(pack(col_q, col_r));
    }

    int n = static_cast<int>(cands.size());
    std::vector<Turn> turns;
    turns.reserve(n * (n - 1) / 2);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            turns.push_back({cands[i], cands[j]});

    return _filter_turns_by_threats(turns);
}

inline std::vector<Turn> MinimaxBot::_generate_threat_turns(
        const flat_set<Coord>& my_threats,
        const flat_set<Coord>& opp_threats) {
    auto [found, wt] = _find_instant_win(_cur_player);
    if (found) return {wt};

    bool maximizing = (_cur_player == _player);

    bool is_a = (_cur_player == P_A);
    bool use_pol = (policy_mode & 4) != 0;
    double sgn = maximizing ? 1.0 : -1.0;
    auto cell_score = [&](Coord c) {
        return use_pol ? _policy_score(pack_q(c), pack_r(c), maximizing)
                       : _move_delta(pack_q(c), pack_r(c), is_a) * sgn;
    };

    std::vector<Coord> opp_cells, my_cells;
    for (Coord c : opp_threats) if (_cand_set.count(c)) opp_cells.push_back(c);
    for (Coord c : my_threats)  if (_cand_set.count(c)) my_cells.push_back(c);

    std::vector<Coord>* primary = nullptr;
    if (!opp_cells.empty())     primary = &opp_cells;
    else if (!my_cells.empty()) primary = &my_cells;
    else return {};

    if (primary->size() >= 2) {
        int n = static_cast<int>(primary->size());
        std::vector<Turn> pairs;
        pairs.reserve(n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                pairs.push_back({(*primary)[i], (*primary)[j]});
        std::sort(pairs.begin(), pairs.end(),
            [&](const Turn& a, const Turn& b) {
                double da = cell_score(a.first) + cell_score(a.second);
                double db = cell_score(b.first) + cell_score(b.second);
                return da > db;
            });
        return pairs;
    }

    Coord tc = (*primary)[0];
    Coord best_comp = tc;
    double best_d = -INF_SCORE;
    for (Coord c : _cand_set) {
        if (c != tc) {
            double d = cell_score(c);
            if (d > best_d) { best_d = d; best_comp = c; }
        }
    }
    if (best_comp == tc) return {};
    return {{coord_min(tc, best_comp), coord_max(tc, best_comp)}};
}

} // namespace opt
