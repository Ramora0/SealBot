/*
 * coord.h -- Coordinate packing and comparison.
 */
#pragma once

#include "constants.h"

// ═══════════════════════════════════════════════════════════════════════
//  Coordinate packing  (still used for Coord values in vectors/turns)
// ═══════════════════════════════════════════════════════════════════════
using Coord = int64_t;

static inline Coord pack(int q, int r) {
    return (static_cast<int64_t>(static_cast<uint32_t>(q)) << 32) |
            static_cast<uint32_t>(r);
}
static inline int pack_q(Coord c) { return static_cast<int32_t>(static_cast<uint32_t>(c >> 32)); }
static inline int pack_r(Coord c) { return static_cast<int32_t>(static_cast<uint32_t>(c)); }

// Lexicographic (q, r) signed comparison via sign-bit flip trick
// (credit: djinnkahn8395)
// Equivalent to:
//   int aq = pack_q(a), ar = pack_r(a), bq = pack_q(b), br = pack_r(b);
//   return (aq < bq) || (aq == bq && ar < br);
static inline bool coord_lt(Coord a, Coord b) {
    return static_cast<uint64_t>(a ^ INT64_C(0x8000000080000000)) <
           static_cast<uint64_t>(b ^ INT64_C(0x8000000080000000));
}
static inline Coord coord_min(Coord a, Coord b) { return coord_lt(a, b) ? a : b; }
static inline Coord coord_max(Coord a, Coord b) { return coord_lt(a, b) ? b : a; }
