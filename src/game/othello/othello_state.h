#ifndef OTHELLOSTATE_H
#define OTHELLOSTATE_H

#include<cstdint>

struct OthelloState {
    uint64_t currentDiscs;
    uint64_t opponentDiscs;
    int moveCount;
};




#endif