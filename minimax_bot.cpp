/*
 * minimax_bot.cpp -- pybind11 wrapper for the minimax engine.
 *
 * Uses baked-in pattern values from pattern_data.h (no external dependencies).
 * Build:  python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"
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
};

PYBIND11_MODULE(minimax_cpp, m) {
    m.doc() = "C++ minimax bot with iterative deepening alpha-beta search";

    py::class_<MinimaxBotWrapper>(m, "MinimaxBot")
        .def(py::init<double>(), py::arg("time_limit") = 0.05)
        .def("get_move", &MinimaxBotWrapper::get_move, py::arg("game"))
        .def("extract_pv", &MinimaxBotWrapper::extract_pv)
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
        .def_property("_nodes",
            [](MinimaxBotWrapper& b) { return b.engine._nodes; },
            [](MinimaxBotWrapper& b, int v) { b.engine._nodes = v; });
}
