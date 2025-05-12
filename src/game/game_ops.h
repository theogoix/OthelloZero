
//contract for GameOps
//this gives a template of what features you have
//to implement for each game you want to add

#ifndef GAMEOPS_H
#define GAMEOPS_H

#include<vector>
#include "game_move.h"
#include "game_state.h"


struct GameState{
    std::vector<GameMove> generateMoves(const GameState& state);
    GameState applyMove(const GameState& state, const GameMove& move);
    const bool isTerminal(const GameState& state);
};

#endif





