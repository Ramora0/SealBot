/*
 * engine.h -- Umbrella include for the SealBot minimax engine.
 *
 * Include this single header to get the full engine.
 * WARNING: This header-only engine assumes a single translation unit.
 */
#pragma once

#include "constants.h"     // includes, constants, hash aliases
#include "coord.h"         // Coord type, pack/unpack, comparisons
#include "engine_types.h"  // Turn, TurnHash, offset structs, TTEntry
#include "containers.h"    // HotSet, CandSet
#include "tables.h"        // direction arrays, Zobrist, precomputed offsets
#include "bot.h"           // MinimaxBot class declaration
#include "board.h"         // make/undo, move delta
#include "movegen.h"       // win/threat detection, turn generation
#include "search.h"        // get_move, extract_pv, minimax, quiescence
#include "vcf.h"           // threat-space forced-win solver
