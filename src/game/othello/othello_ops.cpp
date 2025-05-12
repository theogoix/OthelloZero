#include "othello_ops.h"

std::vector<OthelloMove> OthelloOps::generateMoves(const OthelloState& state){
    std::vector<OthelloMove> move_list = {};

    return move_list;
};

OthelloState OthelloOps::initialState(){
    return {
    0x0000008001000000,
    0x0000000180000000,
    0,
    };
};