/*
 * containers.h -- Array-backed set structures (HotSet, CandSet).
 */
#pragma once

#include "engine_types.h"

// ═══════════════════════════════════════════════════════════════════════
//  Helper structs for array-backed sets
// ═══════════════════════════════════════════════════════════════════════
struct HotEntry { int d, qi, ri; };

struct HotSet {
    bool bits[3][ARR][ARR];
    std::vector<HotEntry> vec;

    void clear() { std::memset(bits, 0, sizeof(bits)); vec.clear(); }

    void insert(int d, int qi, int ri) {
        if (!bits[d][qi][ri]) {
            bits[d][qi][ri] = true;
            vec.push_back({d, qi, ri});
        }
    }

    void erase(int d, int qi, int ri) {
        if (bits[d][qi][ri]) {
            bits[d][qi][ri] = false;
            for (size_t i = 0; i < vec.size(); i++) {
                if (vec[i].d == d && vec[i].qi == qi && vec[i].ri == ri) {
                    vec[i] = vec.back(); vec.pop_back(); break;
                }
            }
        }
    }
};

struct CandSet {
    bool bits[ARR][ARR];
    std::vector<Coord> vec;

    void clear() { std::memset(bits, 0, sizeof(bits)); vec.clear(); }
    bool empty() const { return vec.empty(); }
    size_t size() const { return vec.size(); }
    bool count(Coord c) const { return bits[pack_q(c) + OFF][pack_r(c) + OFF]; }

    void insert(Coord c) {
        int qi = pack_q(c) + OFF, ri = pack_r(c) + OFF;
        if (!bits[qi][ri]) { bits[qi][ri] = true; vec.push_back(c); }
    }

    void erase(Coord c) {
        int qi = pack_q(c) + OFF, ri = pack_r(c) + OFF;
        if (bits[qi][ri]) {
            bits[qi][ri] = false;
            for (size_t i = 0; i < vec.size(); i++) {
                if (vec[i] == c) { vec[i] = vec.back(); vec.pop_back(); break; }
            }
        }
    }

    auto begin() const { return vec.begin(); }
    auto end()   const { return vec.end(); }
};
