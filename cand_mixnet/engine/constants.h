/*
 * constants.h -- Includes, hash container aliases, and engine constants.
 */
#pragma once

#include "../types.h"

// Include the ankerl stl prerequisites directly to avoid "stl.h" path
// collision with pybind11's stl.h on the include path.
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#define ANKERL_UNORDERED_DENSE_STD_MODULE 1
#include "../vendor/ankerl_unordered_dense.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

// ── Alias flat hash containers (still used for TT + history) ──
template <typename K, typename V, typename H = ankerl::unordered_dense::hash<K>>
using flat_map = ankerl::unordered_dense::map<K, V, H>;

template <typename K, typename H = ankerl::unordered_dense::hash<K>>
using flat_set = ankerl::unordered_dense::set<K, H>;

// ═══════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════
static constexpr int    CANDIDATE_CAP      = 15; // 11
static constexpr int    ROOT_CANDIDATE_CAP = 20; // 13
static constexpr int    PAIR_SUM_CAP       = 14;
static constexpr int    NEIGHBOR_DIST      = 2;
static constexpr double DELTA_WEIGHT       = 15; // 1.5
static constexpr int    MAX_QDEPTH         = 16;
static constexpr int    WIN_LENGTH         = 6;
static constexpr double WIN_SCORE          = 100000000.0;
static constexpr double WIN_THRESHOLD      = WIN_SCORE - 1000.0;  // mate-distance detection
static constexpr double INF_SCORE          = std::numeric_limits<double>::infinity();

// Array dimensions -- covers coordinates [-70, 69] with padding for
// windows (+/-5) and neighbor candidates (+/-2).
static constexpr int ARR = 140;
static constexpr int OFF = 70;

// TT flags
static constexpr int8_t TT_EXACT = 0;
static constexpr int8_t TT_LOWER = 1;
static constexpr int8_t TT_UPPER = 2;
