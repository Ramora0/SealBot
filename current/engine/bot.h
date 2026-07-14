/*
 * bot.h -- MinimaxBot class declaration, member data, and inline helpers.
 *
 * Method implementations are split across:
 *   board.h   -- make/undo, move delta
 *   movegen.h -- win/threat detection, turn generation
 *   search.h  -- get_move, extract_pv, minimax, quiescence
 */
#pragma once

#include "containers.h"
#include "tables.h"

// ═══════════════════════════════════════════════════════════════════════
//  MinimaxBot  (namespace opt -- flat-array variant)
// ═══════════════════════════════════════════════════════════════════════
namespace opt {

class MinimaxBot {
public:
    // ── Public attributes ──
    bool   pair_moves = true;
    bool   no_cand_cap = false;
    double time_limit;
    int    last_depth  = 0;
    int    _nodes      = 0;
    double last_score  = 0;
    double last_ebf    = 0;
    int    max_depth   = 200;

    // ── Constructors ──
    MinimaxBot() : time_limit(0.05), _rng(std::random_device{}()),
                   _tt(1 << 20), _tt_mask((1 << 20) - 1) { ensure_tables(); }

    explicit MinimaxBot(double tl)
        : time_limit(tl), _rng(std::random_device{}()),
          _tt(1 << 20), _tt_mask((1 << 20) - 1)
    {
        ensure_tables();
    }

    // ── Pattern loading (call from wrapper after construction) ──
    void load_patterns(const double* values, int count, int eval_length,
                       const std::string& path = "") {
        _pv.assign(values, values + count);
        _eval_length = eval_length;
        _pattern_path_str = path;
        _build_eval_tables();
    }

    void load_patterns(const std::vector<double>& values, int eval_length,
                       const std::string& path = "") {
        load_patterns(values.data(), static_cast<int>(values.size()),
                      eval_length, path);
    }

    // ── Serialisation helpers ──
    EngineState get_state() const {
        return {time_limit, _pv, _eval_length, _pattern_path_str};
    }

    void set_state(const EngineState& es) {
        ensure_tables();
        time_limit = es.time_limit;
        _pv = es.pv;
        _eval_length = es.eval_length;
        _pattern_path_str = es.pattern_path_str;
        _rng = std::mt19937(std::random_device{}());
        _build_eval_tables();
    }

    // ── Public methods (implemented in search.h) ──
    MoveResult get_move(const GameState& gs);
    std::vector<PVStep> extract_pv();

    // ── Check if either player has an instant win ──
    bool has_instant_win() const {
        auto [fa, _a] = _find_instant_win(P_A);
        auto [fb, _b] = _find_instant_win(P_B);
        return fa || fb;
    }

    // ── Threat-space forced-win solver (implemented in vcf.h) ──
    // Returns +1 if `player` (to move, with `stones_left` stones this turn)
    // has a PROVEN forced win within max_turns turns, -1 if provably not
    // via forcing lines, 0 if unknown. If winning, writes the first turn.
    int forced_win(int8_t player, int stones_left, int max_turns, Turn* out);
    int forced_win_position(const GameState& gs, int max_turns, Turn* out);
    // Node cap per forced_win call. 5000 keeps worst-case solve time near
    // 10 ms; raising it to 50000 (~100 ms worst) finds ~8% more deep wins.
    int vcf_node_budget = 5000;
    int vcf_nodes       = 0;       // nodes used by the last forced_win call

    // ── Check near-threat pre-filter (2+ unblocked windows with 3+ stones) ──
    bool has_near_threats() const {
        int a3 = 0, b3 = 0;
        for (int d = 0; d < 3; d++)
            for (int qi = 0; qi < ARR; qi++)
                for (int ri = 0; ri < ARR; ri++) {
                    auto& c = _wc[d][qi][ri];
                    if (c.first >= 3 && c.second == 0) a3++;
                    if (c.second >= 3 && c.first == 0) b3++;
                }
        return a3 >= 2 || b3 >= 2;
    }

private:
    // ── Pattern data ──
    std::vector<double>  _pv;
    int                  _eval_length = 6;
    std::vector<EvalOff> _eval_offsets;
    std::vector<int>     _pow3;
    std::string          _pattern_path_str;

    // ── Board state (flat arrays) ──
    int8_t _board[ARR][ARR] = {};
    std::vector<Coord> _board_cells;

    int8_t _cur_player  = P_A;
    int8_t _moves_left  = 1;
    int8_t _winner      = P_NONE;
    bool   _game_over   = false;
    int    _move_count  = 0;

    // ── 6-cell window counts ──
    std::pair<int8_t,int8_t> _wc[3][ARR][ARR] = {};
    HotSet _hot_a, _hot_b;

    // ── N-cell eval window patterns ──
    int _wp[3][ARR][ARR] = {};

    // ── Candidates ──
    int8_t  _cand_rc[ARR][ARR] = {};
    CandSet _cand_set;
    std::vector<int> _rc_stack;

    // ── Search state ──
    using Clock = std::chrono::steady_clock;
    Clock::time_point _deadline;
    uint64_t _hash      = 0;
    int8_t   _player    = P_A;
    int8_t   _cell_a    = 1;
    int8_t   _cell_b    = 2;
    double   _eval_score = 0;
    int      _ply       = 0;  // distance from root (for mate-distance scoring)

    // Mate-distance TT adjustment: store position-relative win distances
    double _tt_adjust_store(double score) const {
        if (score >  WIN_THRESHOLD) return score + _ply;
        if (score < -WIN_THRESHOLD) return score - _ply;
        return score;
    }
    double _tt_adjust_load(double score) const {
        if (score >  WIN_THRESHOLD) return score - _ply;
        if (score < -WIN_THRESHOLD) return score + _ply;
        return score;
    }

    // ── Transposition table (fixed-size, direct-mapped, always-overwrite) ──
    std::vector<TTEntry> _tt;
    uint64_t _tt_mask = 0;

    TTEntry* _tt_probe(uint64_t full_key) {
        uint32_t verify = static_cast<uint32_t>(full_key >> 32);
        auto& e = _tt[static_cast<size_t>(full_key) & _tt_mask];
        return (e.key == verify) ? &e : nullptr;
    }

    void _tt_store_entry(uint64_t full_key, int depth, double score,
                         int8_t flag, const Turn& move, bool has_move) {
        auto& e    = _tt[static_cast<size_t>(full_key) & _tt_mask];
        uint32_t verify = static_cast<uint32_t>(full_key >> 32);
        // Depth-preferred replacement: keep deeper entries for the same position;
        // always replace if the slot holds a different position.
        if (e.key != verify || depth >= e.depth) {
            e.key      = verify;
            e.depth    = static_cast<int16_t>(depth);
            e.score    = score;
            e.flag     = flag;
            e.move     = move;
            e.has_move = has_move;
        }
    }

    // ── History table ──
    flat_map<Coord, int>        _history;

    // ── Killer moves (2 slots per ply) ──
    static constexpr int MAX_KILLERS_PLY = 64;
    Turn _killers[MAX_KILLERS_PLY][2] = {};

    void _store_killer(int ply, const Turn& t) {
        if (ply >= MAX_KILLERS_PLY) return;
        if (t == _killers[ply][0]) return;
        _killers[ply][1] = _killers[ply][0];
        _killers[ply][0] = t;
    }

    // ── RNG ──
    std::mt19937 _rng;

    // ── Saved state for TimeUp rollback ──
    struct SavedArrays {
        int8_t board[ARR][ARR];
        std::pair<int8_t,int8_t> wc[3][ARR][ARR];
        int wp[3][ARR][ARR];
        int8_t cand_rc[ARR][ARR];
        bool cand_bits[ARR][ARR];
        std::vector<Coord> cand_vec;
        bool hot_a_bits[3][ARR][ARR];
        std::vector<HotEntry> hot_a_vec;
        bool hot_b_bits[3][ARR][ARR];
        std::vector<HotEntry> hot_b_vec;
        std::vector<Coord> board_cells;
    };
    std::unique_ptr<SavedArrays> _saved;

    // ── Inline helpers ──
    inline void _check_time() {
        _nodes++;
        if ((_nodes & 1023) == 0 && Clock::now() >= _deadline)
            throw TimeUp{};
    }

    inline uint64_t _tt_key() const {
        return _hash ^ (static_cast<uint64_t>(_cur_player) * 0x9e3779b97f4a7c15ULL)
                      ^ (static_cast<uint64_t>(_moves_left) * 0x517cc1b727220a95ULL);
    }

    // ── Method declarations (implemented in board.h, movegen.h, search.h) ──
    void _build_eval_tables();
    void _make(int q, int r);
    void _undo(int q, int r, const SavedState& st, int8_t player);
    int  _make_turn(const Turn& turn, UndoStep steps[2]);
    void _undo_turn(const UndoStep steps[], int n);
    double _move_delta(int q, int r, bool is_a) const;

    std::pair<bool, Turn> _find_instant_win(int8_t player) const;
    flat_set<Coord> _find_threat_cells(int8_t player) const;
    std::vector<Turn> _filter_turns_by_threats(const std::vector<Turn>& turns) const;
    std::vector<Turn> _generate_turns();
    std::vector<Turn> _generate_threat_turns(
            const flat_set<Coord>& my_threats,
            const flat_set<Coord>& opp_threats);

    // ── Threat-space solver internals (implemented in vcf.h) ──
    struct VcfState;
    std::unique_ptr<VcfState> _vcf;
    void _vcf_load_position(const GameState& gs);
    int  _vcf_attack(int stones, int remaining, Turn* out);
    int  _vcf_after_attack_turn(int remaining);

    double _quiescence(double alpha, double beta, int qdepth);
    std::pair<Turn, flat_map<Turn, double, TurnHash>>
        _search_root(std::vector<Turn>& turns, int depth);
    double _minimax(int depth, double alpha, double beta);
};

} // namespace opt
