#ifndef OTHELLOGRAPHICS_H
#define OTHELLOGRAPHICS_H

#include "othello_state.h"
#include <string>

namespace Othello{

    const int SECRET_NUMBER = 42;
    void hello_world();
    void print_othello_state(const OthelloState& state, bool black_turn = true);
    OthelloState from_fen(const std::string& fen);

};
#endif