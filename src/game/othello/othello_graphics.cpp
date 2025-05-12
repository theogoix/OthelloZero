#include <iostream>
#include "othello_graphics.h"

void hello_world() {
    std::cout << "hello world, and goodbye";
};

void print_othello_state(const OthelloState& state){
    char black_symbol = 'X';
    char white_symbol = 'O';
    bool black_turn = state.moveCount % 2;
    uint64_t black_discs = black_turn? state.currentDiscs : state.opponentDiscs;
    uint64_t white_discs = black_turn ? state.opponentDiscs : state.currentDiscs;
    std::cout << black_discs << "\n" << white_discs << "\n";
};