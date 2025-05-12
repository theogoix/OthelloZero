#ifndef OTHELLOOPS_H
#define OTHELLOOPS_H

#include<vector>
#include "othello_state.h"
#include "othello_move.h"

struct OthelloOps {
    static std::vector<OthelloMove> generateMoves(const OthelloState& state);
    static OthelloState applyMove(const OthelloState& state, const OthelloMove& move);
    static bool isTerminal(const OthelloState& state);
    static OthelloState initialState();
};


#endif