/*
 * bot.h -- MinimaxBot class declaration, member data, and inline helpers.
 *
 * Method implementations are split across:
 *   board.h   -- make/undo, move delta
 *   movegen.h -- win/threat detection, turn generation
 *   search.h  -- get_move, extract_pv, minimax, quiescence
 */
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include "containers.h"
#include "tables.h"
#include "../codebook_data.h"
#include "../net_data.h"
#include "../policy_data.h"
#include "../trunk_data.h"

// ═══════════════════════════════════════════════════════════════════════
//  MinimaxBot  (namespace opt -- flat-array variant)
// ═══════════════════════════════════════════════════════════════════════
namespace opt {

class MinimaxBot {
public:
    // ── Public attributes ──
    bool   pair_moves = true;
    bool   no_cand_cap = false;
    // Candidate caps; overridable via SEAL_CAND_CAP / SEAL_ROOT_CAP env vars
    // (read in the constructor) so gate harnesses can sweep without rebuilds.
    int    cand_cap      = CANDIDATE_CAP;
    int    root_cand_cap = ROOT_CANDIDATE_CAP;
    // Tactical safety net: always keep the top-K cells by |linear delta| in
    // the candidate set even when the policy ranks them below the cap
    // (policy tail can drop forced blocks). SEAL_DELTA_KEEP overrides.
    int    delta_keep    = 0;
    // Ordering source bits: 1=our-interior, 2=root, 4=threat path,
    // 8=opp-interior, 32=policy reorders delta picks, 64=rank fusion at
    // interior (min(policy_rank, delta_rank): coverage = union of both).
    // Default 74 = policy at root + opp-interior fusion: +402 vs no-policy,
    // equal to root-only head-to-head, strictly wider refutation coverage.
    // SEAL_POLICY_MODE overrides.
    int    policy_mode   = 74;
    // Lazy SMP: helper threads sharing the TT (SEAL_THREADS, default 1).
    int    threads       = 1;
    double time_limit;
    int    last_depth  = 0;
    int    _nodes      = 0;
    double last_score  = 0;
    double last_ebf    = 0;
    int    max_depth   = 200;

    // ── Constructors ──
    MinimaxBot() : time_limit(0.05), _rng(std::random_device{}()),
                   _tt(std::make_shared<std::vector<TTEntry>>(1 << 20)),
                   _tt_mask((1 << 20) - 1) { ensure_tables(); }

    explicit MinimaxBot(double tl)
        : time_limit(tl), _rng(std::random_device{}()),
          _tt(std::make_shared<std::vector<TTEntry>>(1 << 20)),
          _tt_mask((1 << 20) - 1)
    {
        ensure_tables();
        if (const char* e = std::getenv("SEAL_THREADS"))
            threads = std::max(1, std::atoi(e));
        if (const char* e = std::getenv("SEAL_CAND_CAP"))
            cand_cap = std::atoi(e);
        if (const char* e = std::getenv("SEAL_ROOT_CAP"))
            root_cand_cap = std::atoi(e);
        if (const char* e = std::getenv("SEAL_DELTA_KEEP"))
            delta_keep = std::atoi(e);
        if (const char* e = std::getenv("SEAL_POLICY_MODE"))
            policy_mode = std::atoi(e);
        if (const char* e = std::getenv("SEAL_EVAL"))
            _use_trunk = (std::string(e) == std::string("trunk"));
        if (const char* e = std::getenv("SEAL_TRUNK_BLEND"))
            _trunk_blend = std::atof(e);
        // Policy source, independent of the value head: 1 = trunk MLP,
        // 0 = legacy PW/PC tables. Defaults to following SEAL_EVAL.
        _trunk_policy = _use_trunk;
        if (const char* e = std::getenv("SEAL_TRUNK_POLICY"))
            _trunk_policy = std::atoi(e) != 0;
        if ((_use_trunk || _trunk_policy) && !_load_eraw()) {
            _use_trunk = false;
            _trunk_policy = false;
        }
        _need_acc2 = _use_trunk || _trunk_policy;
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

    // ── Debug / training hooks (implemented in board.h) ──
    double static_eval(const GameState& gs);
    double policy_score_debug(int q, int r, bool for_root) const {
        return _policy_score(q, r, for_root);
    }
    std::pair<std::vector<std::pair<int,int>>, std::vector<std::pair<int,int>>>
        debug_features(const GameState& gs);
    std::vector<float> get_acc() const {
        if (_use_trunk)
            return std::vector<float>(_acc2, _acc2 + TRK_K);
        return std::vector<float>(_acc, _acc + NET_K);
    }

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

    // ── NNUE state: accumulator + 11-cell line patterns per (cell, dir) ──
    float   _acc[NET_K] = {};
    int     _lp[3][ARR][ARR] = {};
    uint8_t _lc[3][ARR][ARR] = {};  // cached LINE_CODEBOOK[_lp[d][q][r]]

    // ── Trunk (cellnl) state: SEAL_EVAL=trunk switches leaf eval + policy
    // to the shared-trunk net. acc2 = sum of clamped per-cell activations
    // (raw 3^11 line embeddings summed over 3 dirs) + EW bag. The E_raw
    // table (3^11 x K floats, ~23 MB) is loaded from TRK_ERAW_PATH at
    // construction (SEAL_TRUNK_BLOB overrides), shared across instances. ──
    bool   _use_trunk = false;
    bool   _trunk_policy = false;
    bool   _need_acc2 = false;   // maintain _acc2 (trunk value OR policy)
    float  _trunk_blend = TRK_LIN_BLEND;
    float  _acc2[TRK_K] = {};
    inline static std::vector<float> g_eraw;

    static bool _load_eraw() {
        if (!g_eraw.empty()) return true;
        const char* path = std::getenv("SEAL_TRUNK_BLOB");
        if (!path) path = TRK_ERAW_PATH;
        std::FILE* f = std::fopen(path, "rb");
        if (!f) {
            std::fprintf(stderr, "trunk: cannot open %s, falling back\n",
                         path);
            return false;
        }
        g_eraw.resize(static_cast<size_t>(TRK_ERAW_ROWS) * TRK_K);
        size_t got = std::fread(g_eraw.data(), sizeof(float),
                                g_eraw.size(), f);
        std::fclose(f);
        if (got != g_eraw.size()) {
            std::fprintf(stderr, "trunk: short read, falling back\n");
            g_eraw.clear();
            return false;
        }
        return true;
    }

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

    // ── Transposition table (fixed-size, direct-mapped, SHARED across
    // lazy-SMP helpers via shared_ptr; lock-free XOR-consistent entries) ──
    std::shared_ptr<std::vector<TTEntry>> _tt;
    uint64_t _tt_mask = 0;

    bool _tt_probe(uint64_t full_key, TTView& out) const {
        const TTEntry& e = (*_tt)[static_cast<size_t>(full_key) & _tt_mask];
        uint64_t b = e.b, c = e.c;
        uint64_t meta = e.a ^ b ^ c;
        if (static_cast<uint32_t>(meta) !=
            static_cast<uint32_t>(full_key >> 32))
            return false;
        out.depth    = static_cast<int16_t>(meta >> 32);
        out.flag     = static_cast<int8_t>((meta >> 48) & 0xff);
        out.has_move = ((meta >> 56) & 1) != 0;
        std::memcpy(&out.score, &b, 8);
        int8_t q1 = static_cast<int8_t>(c & 0xff);
        int8_t r1 = static_cast<int8_t>((c >> 8) & 0xff);
        int8_t q2 = static_cast<int8_t>((c >> 16) & 0xff);
        int8_t r2 = static_cast<int8_t>((c >> 24) & 0xff);
        out.move = {pack(q1, r1), pack(q2, r2)};
        return true;
    }

    void _tt_store_entry(uint64_t full_key, int depth, double score,
                         int8_t flag, const Turn& move, bool has_move) {
        TTEntry& e = (*_tt)[static_cast<size_t>(full_key) & _tt_mask];
        uint64_t b0 = e.b, c0 = e.c;
        uint64_t meta0 = e.a ^ b0 ^ c0;
        uint32_t verify = static_cast<uint32_t>(full_key >> 32);
        // Depth-preferred replacement: keep deeper entries for the same
        // position; always replace a different position.
        if (static_cast<uint32_t>(meta0) == verify &&
            depth < static_cast<int16_t>(meta0 >> 32))
            return;
        uint64_t b;
        std::memcpy(&b, &score, 8);
        uint64_t c =
            (static_cast<uint64_t>(static_cast<uint8_t>(
                 static_cast<int8_t>(pack_q(move.first))))) |
            (static_cast<uint64_t>(static_cast<uint8_t>(
                 static_cast<int8_t>(pack_r(move.first)))) << 8) |
            (static_cast<uint64_t>(static_cast<uint8_t>(
                 static_cast<int8_t>(pack_q(move.second)))) << 16) |
            (static_cast<uint64_t>(static_cast<uint8_t>(
                 static_cast<int8_t>(pack_r(move.second)))) << 24);
        uint64_t meta = static_cast<uint64_t>(verify) |
            (static_cast<uint64_t>(static_cast<uint16_t>(
                 static_cast<int16_t>(depth))) << 32) |
            (static_cast<uint64_t>(static_cast<uint8_t>(flag)) << 48) |
            (static_cast<uint64_t>(has_move ? 1 : 0) << 56);
        e.b = b;
        e.c = c;
        e.a = meta ^ b ^ c;
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

    // ── Partial root-iteration harvest (see _search_root) ──
    Turn   _root_partial_best{};
    double _root_partial_score = 0;
    bool   _root_partial_valid = false;

    // ── Lazy SMP state ──
    std::vector<std::unique_ptr<MinimaxBot>> _helpers;
    std::atomic<bool>* _stop_ext = nullptr;  // set on helpers by the main bot

    // Copy search-relevant config into a helper (tables/weights shared).
    void _clone_config_from(const MinimaxBot& m) {
        _pv          = m._pv;
        _eval_length = m._eval_length;
        _build_eval_tables();
        cand_cap      = m.cand_cap;
        root_cand_cap = m.root_cand_cap;
        delta_keep    = m.delta_keep;
        policy_mode   = m.policy_mode;
        pair_moves    = m.pair_moves;
        no_cand_cap   = m.no_cand_cap;
        max_depth     = m.max_depth;
        _use_trunk    = m._use_trunk;
        _trunk_policy = m._trunk_policy;
        _need_acc2    = m._need_acc2;
        _trunk_blend  = m._trunk_blend;
        _tt           = m._tt;      // share the table
        _tt_mask      = m._tt_mask;
    }

    // ── Saved state for TimeUp rollback ──
    struct SavedArrays {
        int8_t board[ARR][ARR];
        std::pair<int8_t,int8_t> wc[3][ARR][ARR];
        int wp[3][ARR][ARR];
        float acc[NET_K];
        float acc2[TRK_K];
        int lp[3][ARR][ARR];
        uint8_t lc[3][ARR][ARR];
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
        // Trunk nodes are ~1.5x slower; check the clock more often so the
        // overshoot past the deadline stays comparable to the legacy net.
        int mask = _use_trunk ? 255 : 1023;
        if ((_nodes & mask) == 0) {
            if (_stop_ext && _stop_ext->load(std::memory_order_relaxed))
                throw TimeUp{};
            if (Clock::now() >= _deadline)
                throw TimeUp{};
        }
    }

    inline uint64_t _tt_key() const {
        return _hash ^ (static_cast<uint64_t>(_cur_player) * 0x9e3779b97f4a7c15ULL)
                      ^ (static_cast<uint64_t>(_moves_left) * 0x517cc1b727220a95ULL);
    }

    // ── NNUE inline helpers ──

    // Conjunction class of a cell, or -1 if no stones within line range.
    inline int _conj_class(int qi, int ri) const {
        int lp0 = _lp[0][qi][ri], lp1 = _lp[1][qi][ri], lp2 = _lp[2][qi][ri];
        if ((lp0 | lp1 | lp2) == 0) return -1;
        uint8_t c0 = _lc[0][qi][ri];
        uint8_t c1 = _lc[1][qi][ri];
        uint8_t c2 = _lc[2][qi][ri];
        int8_t b = _board[qi][ri];
        if (b == 0) {
            int p0 = (c0 & 7) * 6 + (c0 >> 3);
            int p1 = (c1 & 7) * 6 + (c1 >> 3);
            int p2 = (c2 & 7) * 6 + (c2 >> 3);
            return CANON_EMPTY[(p0 * 36 + p1) * 36 + p2];
        }
        int digit = (b == P_A) ? _cell_a : _cell_b;  // 1 = root player
        int o0 = (digit == 1) ? (c0 & 7) : (c0 >> 3);
        int o1 = (digit == 1) ? (c1 & 7) : (c1 >> 3);
        int o2 = (digit == 1) ? (c2 & 7) : (c2 >> 3);
        return CONJ_EMPTY_CLASSES + (digit == 2) * CONJ_OCC_RANKS
               + CANON_OCC[(o0 * 6 + o1) * 6 + o2];
    }

    // ── Trunk inline helpers ──

    inline const float* _eraw_row(int code) const {
        return g_eraw.data() + static_cast<size_t>(code) * TRK_K;
    }

    // Add (sgn=+1) or remove (sgn=-1) the clamped activation of one cell.
    inline void _trunk_cell(int qi, int ri, float sgn) {
        int l0 = _lp[0][qi][ri], l1 = _lp[1][qi][ri], l2 = _lp[2][qi][ri];
        if ((l0 | l1 | l2) == 0) return;
        const float* e0 = _eraw_row(l0);
        const float* e1 = _eraw_row(l1);
        const float* e2 = _eraw_row(l2);
        for (int k = 0; k < TRK_K; k++) {
            float s = e0[k] + e1[k] + e2[k];
            s = s < 0.f ? 0.f : (s > TRK_CLIP ? TRK_CLIP : s);
            _acc2[k] += sgn * s;
        }
    }

    inline void _trunk_cells(const int* cq, const int* cr, int n, float sgn) {
        for (int i = 0; i < n; i++) _trunk_cell(cq[i], cr[i], sgn);
    }

    inline double _leaf_eval_trunk() const {
        float h[TRK_K];
        for (int k = 0; k < TRK_K; k++) {
            float v = _acc2[k];
            h[k] = v < 0.f ? 0.f : (v > TRK_CLIP ? TRK_CLIP : v);
        }
        float g0 = static_cast<float>(_move_count) * 0.02f;
        float g1 = (_cur_player == _player ? 1.0f : -1.0f)
                   * static_cast<float>(_moves_left) * 0.5f;
        float out = TRK_B2;
        for (int j = 0; j < TRK_H; j++) {
            float s = TRK_B1[j];
            for (int k = 0; k < TRK_K; k++) s += TRK_W1[j][k] * h[k];
            s += TRK_W1[j][TRK_K] * g0 + TRK_W1[j][TRK_K + 1] * g1;
            if (s > 0.f) out += TRK_W2[j] * s;
        }
        return static_cast<double>(out) * TRK_OUT_SCALE
               + _trunk_blend * _eval_score;
    }

    // Trunk policy head: MLP([clamp(cell act); clamp(acc2); g0; g1]).
    // Mirror-trained: g1 sign carries who places (for_root => digit 1).
    inline double _policy_score_trunk(int q, int r, bool for_root) const {
        int qi = q + OFF, ri = r + OFF;
        float x[2 * TRK_K + 2];
        int l0 = _lp[0][qi][ri], l1 = _lp[1][qi][ri], l2 = _lp[2][qi][ri];
        // NB: all-zero lines still read row 0 — the trained head expects
        // clamp(3*eraw[0]) there, not zeros (parity bug otherwise).
        const float* e0 = _eraw_row(l0);
        const float* e1 = _eraw_row(l1);
        const float* e2 = _eraw_row(l2);
        for (int k = 0; k < TRK_K; k++) {
            float s = e0[k] + e1[k] + e2[k];
            x[k] = s < 0.f ? 0.f : (s > TRK_CLIP ? TRK_CLIP : s);
        }
        for (int k = 0; k < TRK_K; k++) {
            float v = _acc2[k];
            x[TRK_K + k] = v < 0.f ? 0.f : (v > TRK_CLIP ? TRK_CLIP : v);
        }
        x[2 * TRK_K]     = static_cast<float>(_move_count) * 0.02f;
        x[2 * TRK_K + 1] = (for_root ? 1.0f : -1.0f)
                           * static_cast<float>(_moves_left) * 0.5f;
        float out = TRK_P2B;
        for (int j = 0; j < TRK_HP; j++) {
            float s = TRK_P1B[j];
            for (int k = 0; k < 2 * TRK_K + 2; k++)
                s += TRK_P1[j][k] * x[k];
            if (s > 0.f) out += TRK_P2[j] * s;
        }
        return static_cast<double>(out);
    }

    // Strix-distilled ordering score for placing on empty (q, r). Tables
    // are mover-relative; _wp and classes are root-relative, so opponent
    // moves read through the color-mirror index tables. Higher = better
    // for the side placing the stone.
    inline double _policy_score(int q, int r, bool for_root) const {
        if (_trunk_policy) return _policy_score_trunk(q, r, for_root);
        int qi = q + OFF, ri = r + OFF;
        double s = 0.0;
        if (for_root) {
            for (const auto& eo : _eval_offsets)
                s += POLICY_W[_wp[eo.d_idx][qi - eo.oq][ri - eo.or_]];
            int cls = _conj_class(qi, ri);
            s += POLICY_C[cls >= 0 ? cls : CONJ_NUM_CLASSES];
        } else {
            for (const auto& eo : _eval_offsets)
                s += POLICY_W[POLICY_MIR_W[_wp[eo.d_idx][qi - eo.oq][ri - eo.or_]]];
            int cls = _conj_class(qi, ri);
            s += POLICY_C[cls >= 0 ? POLICY_MIR_C[cls] : CONJ_NUM_CLASSES];
        }
        return s;
    }

    // Policy-ordered candidate selection with a linear-delta safety net:
    // sort by policy (desc), cap, then force-in the top delta_keep cells by
    // |move delta| that the cap dropped. Writes survivors into `cands`.
    inline void _select_candidates(std::vector<Coord>& cands, int cap,
                                   bool maximizing, bool is_a,
                                   bool use_policy) {
        std::vector<std::pair<double, Coord>> scored;
        scored.reserve(cands.size());
        if (use_policy && (policy_mode & 64)) {
            // Rank fusion: key = min(policy_rank, delta_rank). Coverage is
            // the UNION of both systems' recall (tree at least as sound as
            // delta's), top ordering is the policy's (tiebreak by policy).
            size_t n = cands.size();
            std::vector<std::pair<double, int>> ps(n), ds(n);
            double sgn = maximizing ? 1.0 : -1.0;
            for (size_t k = 0; k < n; k++) {
                Coord c = cands[k];
                ps[k] = {-_policy_score(pack_q(c), pack_r(c), maximizing),
                         static_cast<int>(k)};
                ds[k] = {-_move_delta(pack_q(c), pack_r(c), is_a) * sgn,
                         static_cast<int>(k)};
            }
            std::sort(ps.begin(), ps.end());
            std::sort(ds.begin(), ds.end());
            std::vector<int> rp(n), rd(n);
            for (size_t r = 0; r < n; r++) {
                rp[ps[r].second] = static_cast<int>(r);
                rd[ds[r].second] = static_cast<int>(r);
            }
            for (size_t k = 0; k < n; k++) {
                double key = std::min(rp[k], rd[k]) * 1000.0 + rp[k];
                scored.push_back({-key, cands[k]});
            }
        } else if (use_policy) {
            for (Coord c : cands)
                scored.push_back({_policy_score(pack_q(c), pack_r(c), maximizing), c});
        } else {
            double sgn = maximizing ? 1.0 : -1.0;
            for (Coord c : cands)
                scored.push_back({_move_delta(pack_q(c), pack_r(c), is_a) * sgn, c});
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        int keep = std::min(static_cast<int>(scored.size()), cap);
        cands.clear();
        for (int i = 0; i < keep; i++)
            cands.push_back(scored[i].second);
        // bit 5: policy REORDERS the delta-selected survivors (coverage
        // from delta, cutoff ordering from the net).
        if (!use_policy && (policy_mode & 32)) {
            std::sort(cands.begin(), cands.end(), [&](Coord a, Coord b) {
                double pa = _policy_score(pack_q(a), pack_r(a), maximizing);
                double pb = _policy_score(pack_q(b), pack_r(b), maximizing);
                if (pa != pb) return pa > pb;
                return a < b;
            });
        }
        // Threat-first partition: interior pair generation only emits pairs
        // with index-sum <= PAIR_SUM_CAP, so refutations must sit at the
        // FRONT or their pairs never exist. Stable-partition must-block
        // cells (opponent's >=4-window empties) ahead of the rest.
        if (use_policy) {
            int8_t opp = (_cur_player == P_A) ? P_B : P_A;
            auto threats = _find_threat_cells(opp);
            if (!threats.empty())
                std::stable_partition(cands.begin(), cands.end(),
                    [&](Coord c) { return threats.count(c) != 0; });
        }
        if (delta_keep > 0 && keep < static_cast<int>(scored.size())) {
            // Append top delta_keep dropped cells by the OTHER scorer:
            // tactical |delta| when policy selected, policy score when
            // delta selected (union widening — adds coverage, never drops).
            std::vector<std::pair<double, Coord>> tail;
            tail.reserve(scored.size() - keep);
            for (size_t i = keep; i < scored.size(); i++) {
                Coord c = scored[i].second;
                double s = use_policy
                    ? std::abs(_move_delta(pack_q(c), pack_r(c), is_a))
                    : _policy_score(pack_q(c), pack_r(c), maximizing);
                tail.push_back({s, c});
            }
            int extra = std::min(delta_keep, static_cast<int>(tail.size()));
            std::partial_sort(tail.begin(), tail.begin() + extra, tail.end(),
                              [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            for (int i = 0; i < extra; i++)
                cands.push_back(tail[i].second);
        }
    }

    // The 31 cells whose conjunction class can change when (qi, ri) flips.
    inline int _collect_conj_cells(int qi, int ri, int* cq, int* cr) const {
        int n = 0;
        cq[n] = qi; cr[n] = ri; n++;
        for (int d = 0; d < 3; d++)
            for (int m = -LP_CENTER; m <= LP_CENTER; m++) {
                if (m == 0) continue;
                cq[n] = qi + m * DIR_Q[d];
                cr[n] = ri + m * DIR_R[d];
                n++;
            }
        return n;
    }

    // Snapshot classes before mutation; apply only the diffs after.
    inline void _conj_snapshot(const int* cq, const int* cr, int n,
                               int* out_cls) const {
        for (int i = 0; i < n; i++)
            out_cls[i] = _conj_class(cq[i], cr[i]);
    }

    inline void _conj_diff_apply(const int* cq, const int* cr, int n,
                                 const int* old_cls) {
        for (int i = 0; i < n; i++) {
            int nc = _conj_class(cq[i], cr[i]);
            int oc = old_cls[i];
            if (nc == oc) continue;
            if (oc >= 0) {
                const float* e = NET_EC[oc];
                for (int k = 0; k < NET_K; k++) _acc[k] -= e[k];
            }
            if (nc >= 0) {
                const float* e = NET_EC[nc];
                for (int k = 0; k < NET_K; k++) _acc[k] += e[k];
            }
        }
    }

    inline double _leaf_eval() const {
        if (_use_trunk) return _leaf_eval_trunk();
        float h[NET_K];
        for (int k = 0; k < NET_K; k++) {
            float v = _acc[k];
            h[k] = v < 0.f ? 0.f : (v > NET_CLIP ? NET_CLIP : v);
        }
        float g0 = static_cast<float>(_move_count) * 0.02f;
        // tempo: +moves_left/2 when the root player is to move, else negated
        float g1 = (_cur_player == _player ? 1.0f : -1.0f)
                   * static_cast<float>(_moves_left) * 0.5f;
        float out = NET_B2;
        for (int j = 0; j < NET_H; j++) {
            float s = NET_B1[j];
            for (int k = 0; k < NET_K; k++) s += NET_W1[j][k] * h[k];
            s += NET_W1[j][NET_K] * g0 + NET_W1[j][NET_K + 1] * g1;
            if (s > 0.f) out += NET_W2[j] * s;
        }
        // Hybrid: net (global judgment) + scaled linear window sum
        // (local gradient; already incrementally maintained for ordering).
        return static_cast<double>(out) * NET_OUT_SCALE
               + NET_LIN_BLEND * _eval_score;
    }

    // ── Method declarations (implemented in board.h, movegen.h, search.h) ──
    void _build_eval_tables();
    void _load_position(const GameState& gs);
    void _init_eval_arrays();
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

    void _setup_position(const GameState& gs);
    void _helper_loop(GameState gs, Clock::time_point deadline,
                      std::atomic<bool>* stop, int offset);
};

} // namespace opt
