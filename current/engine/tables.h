/*
 * tables.h -- Direction arrays, Zobrist hashing, precomputed offset tables.
 */
#pragma once

#include "engine_types.h"

// ═══════════════════════════════════════════════════════════════════════
//  Direction arrays
// ═══════════════════════════════════════════════════════════════════════
static constexpr int DIR_Q[3] = {1, 0, 1};
static constexpr int DIR_R[3] = {0, 1, -1};
static constexpr int COLONY_DQ[6] = { 1, -1,  0,  0,  1, -1};
static constexpr int COLONY_DR[6] = { 0,  0,  1, -1, -1,  1};

// ═══════════════════════════════════════════════════════════════════════
//  Precomputed offset tables (initialised once)
// ═══════════════════════════════════════════════════════════════════════
static std::vector<WinOff> g_win_offsets;
static std::vector<NbOff>  g_nb_offsets;
static std::vector<std::pair<int,int>> g_inner_pairs;

static inline int hex_distance(int dq, int dr) {
    return std::max({std::abs(dq), std::abs(dr), std::abs(dq + dr)});
}

// Powers of 3 for the 11-cell line-pattern index (conjunction features)
static constexpr int POW3_11[11] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561,
                                    19683, 59049};

// Zobrist tables -- flat arrays, deterministic per (q, r) via splitmix64.
static uint64_t g_zobrist_a[ARR][ARR];
static uint64_t g_zobrist_b[ARR][ARR];

static inline uint64_t splitmix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

static inline uint64_t get_zobrist(int q, int r, int8_t player) {
    return (player == P_A) ? g_zobrist_a[q + OFF][r + OFF]
                           : g_zobrist_b[q + OFF][r + OFF];
}

static bool g_tables_ready = false;
static void ensure_tables() {
    if (g_tables_ready) return;
    for (int d = 0; d < 3; d++)
        for (int k = 0; k < WIN_LENGTH; k++)
            g_win_offsets.push_back({d, k * DIR_Q[d], k * DIR_R[d]});
    for (int dq = -NEIGHBOR_DIST; dq <= NEIGHBOR_DIST; dq++)
        for (int dr = -NEIGHBOR_DIST; dr <= NEIGHBOR_DIST; dr++)
            if ((dq || dr) && hex_distance(dq, dr) <= NEIGHBOR_DIST)
                g_nb_offsets.push_back({dq, dr});
    for (int i = 0; i < ARR; i++)
        for (int j = 0; j < ARR; j++) {
            int q = i - OFF, r = j - OFF;
            uint64_t base = static_cast<uint64_t>(static_cast<uint32_t>(q)) << 32
                          | static_cast<uint64_t>(static_cast<uint32_t>(r));
            g_zobrist_a[i][j] = splitmix64(base ^ 0xa02bdbf7bb3c0195ULL);
            g_zobrist_b[i][j] = splitmix64(base ^ 0x3f84d5b5b5470917ULL);
        }
    for (int i = 0; i < CANDIDATE_CAP; i++)
        for (int j = i + 1; j < CANDIDATE_CAP && i + j <= PAIR_SUM_CAP; j++)
            g_inner_pairs.push_back({i, j});
    g_tables_ready = true;
}
