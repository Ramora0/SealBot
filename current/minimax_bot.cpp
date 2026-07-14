/*
 * minimax_bot.cpp -- pybind11 wrapper for the minimax engine.
 *
 * Uses baked-in pattern values from pattern_data.h (no external dependencies).
 * Build:  python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine/engine.h"
#include "pattern_data.h"

namespace py = pybind11;

static GameState extract_game_state(py::object game) {
    py::module_ game_mod = py::module_::import("game");
    py::object  PyA = game_mod.attr("Player").attr("A");

    py::dict py_board = game.attr("board").cast<py::dict>();

    GameState gs;
    gs.cells.reserve(py_board.size());
    for (auto item : py_board) {
        py::tuple key = item.first.cast<py::tuple>();
        int q = key[0].cast<int>(), r = key[1].cast<int>();
        int8_t p = item.second.is(PyA) ? P_A : P_B;
        gs.cells.push_back({q, r, p});
    }

    py::object py_cur = game.attr("current_player");
    gs.cur_player  = py_cur.is(PyA) ? P_A : P_B;
    gs.moves_left  = game.attr("moves_left_in_turn").cast<int8_t>();
    gs.move_count  = game.attr("move_count").cast<int>();
    return gs;
}

struct MinimaxBotWrapper {
    opt::MinimaxBot engine;

    MinimaxBotWrapper(double tl = 0.05) : engine(tl) {
        std::vector<double> pv(PATTERN_VALUES, PATTERN_VALUES + PATTERN_COUNT);
        engine.load_patterns(pv, PATTERN_EVAL_LENGTH);
    }

    py::list get_move(py::object game) {
        auto gs = extract_game_state(game);
        if (gs.cells.empty()) {
            py::list res;
            res.append(py::make_tuple(0, 0));
            return res;
        }

        auto mr = engine.get_move(gs);

        py::list res;
        res.append(py::make_tuple(mr.q1, mr.r1));
        if (mr.num_moves > 1)
            res.append(py::make_tuple(mr.q2, mr.r2));
        return res;
    }

    // Threat-space forced-win probe. Returns (result, moves) where result is
    // +1 proven win / -1 provably no forcing win / 0 unknown, and moves is
    // the winning first turn as [(q, r), ...] when result == +1.
    py::tuple forced_win(py::object game, int max_turns) {
        auto gs = extract_game_state(game);
        Turn t{};
        int r = engine.forced_win_position(gs, max_turns, &t);
        py::list moves;
        if (r == 1) {
            moves.append(py::make_tuple(pack_q(t.first), pack_r(t.first)));
            if (gs.moves_left >= 2 && t.second != t.first)
                moves.append(py::make_tuple(pack_q(t.second), pack_r(t.second)));
        }
        return py::make_tuple(r, moves);
    }

    py::list extract_pv() {
        auto pv = engine.extract_pv();
        py::list result;
        for (const auto& step : pv) {
            py::dict d;
            d["player"] = (step.player == P_A) ? "A" : "B";
            py::list moves;
            for (const auto& [q, r] : step.moves)
                moves.append(py::make_tuple(q, r));
            d["moves"] = moves;
            result.append(d);
        }
        return result;
    }

    double eval_position(py::object game) {
        return engine.static_eval(extract_game_state(game));
    }

    // Max |acc drift| after a search: exercises make/undo + rollback, then
    // compares the incrementally-maintained accumulator to a fresh recompute.
    double acc_drift(py::object game, double tl) {
        auto gs = extract_game_state(game);
        double tl_save = engine.time_limit;
        engine.time_limit = tl;
        engine.get_move(gs);
        engine.time_limit = tl_save;
        auto after_search = engine.get_acc();
        engine.static_eval(gs);  // fresh recompute of the same position
        auto fresh = engine.get_acc();
        double worst = 0.0;
        for (size_t k = 0; k < fresh.size(); k++)
            worst = std::max(worst, std::abs(
                static_cast<double>(after_search[k]) - fresh[k]));
        return worst;
    }

    py::list policy_debug(py::object game, py::list cells, bool for_root) {
        engine.static_eval(extract_game_state(game));
        py::list out;
        for (auto item : cells) {
            auto t = item.cast<py::tuple>();
            out.append(engine.policy_score_debug(
                t[0].cast<int>(), t[1].cast<int>(), for_root));
        }
        return out;
    }

    py::dict feature_counts(py::object game) {
        auto [wf, cf] = engine.debug_features(extract_game_state(game));
        py::dict w, c;
        for (auto& [k, v] : wf) w[py::int_(k)] = v;
        for (auto& [k, v] : cf) c[py::int_(k)] = v;
        py::dict out;
        out["w"] = w;
        out["c"] = c;
        return out;
    }
};

PYBIND11_MODULE(minimax_cpp, m) {
    m.doc() = "C++ minimax bot with iterative deepening alpha-beta search";

    py::class_<MinimaxBotWrapper>(m, "MinimaxBot")
        .def(py::init<double>(), py::arg("time_limit") = 0.05)
        .def("get_move", &MinimaxBotWrapper::get_move, py::arg("game"))
        .def("extract_pv", &MinimaxBotWrapper::extract_pv)
        .def("eval_position", &MinimaxBotWrapper::eval_position, py::arg("game"))
        .def("feature_counts", &MinimaxBotWrapper::feature_counts, py::arg("game"))
        .def("policy_debug", &MinimaxBotWrapper::policy_debug,
             py::arg("game"), py::arg("cells"), py::arg("for_root") = true)
        .def("acc_drift", &MinimaxBotWrapper::acc_drift,
             py::arg("game"), py::arg("tl") = 0.05)
        .def("forced_win", &MinimaxBotWrapper::forced_win,
             py::arg("game"), py::arg("max_turns") = 8)
        .def_property("vcf_node_budget",
            [](MinimaxBotWrapper& b) { return b.engine.vcf_node_budget; },
            [](MinimaxBotWrapper& b, int v) { b.engine.vcf_node_budget = v; })
        .def_property_readonly("vcf_nodes",
            [](MinimaxBotWrapper& b) { return b.engine.vcf_nodes; })
        .def_property_readonly("work_nodes",
            [](MinimaxBotWrapper& b) {
                return static_cast<long long>(b.engine._nodes)
                       + b.engine.vcf_work;
            })
        .def("__str__", [](const MinimaxBotWrapper&) { return "SealBot"; })
        .def_property("time_limit",
            [](MinimaxBotWrapper& b) { return b.engine.time_limit; },
            [](MinimaxBotWrapper& b, double v) { b.engine.time_limit = v; })
        .def_property("last_depth",
            [](MinimaxBotWrapper& b) { return b.engine.last_depth; },
            [](MinimaxBotWrapper& b, int v) { b.engine.last_depth = v; })
        .def_property("last_score",
            [](MinimaxBotWrapper& b) { return b.engine.last_score; },
            [](MinimaxBotWrapper& b, double v) { b.engine.last_score = v; })
        .def_property_readonly("pair_moves",
            [](MinimaxBotWrapper& b) { return b.engine.pair_moves; })
        .def_property("max_depth",
            [](MinimaxBotWrapper& b) { return b.engine.max_depth; },
            [](MinimaxBotWrapper& b, int v) { b.engine.max_depth = v; })
        .def_property("_nodes",
            [](MinimaxBotWrapper& b) { return b.engine._nodes; },
            [](MinimaxBotWrapper& b, int v) { b.engine._nodes = v; });
}
