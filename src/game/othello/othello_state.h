#ifndef OTHELLOSTATE_H
#define OTHELLOSTATE_H

#include<cstdint>

namespace Othello{
    
    const int NCOL = 8;
    const int NROW = 8;
    

    struct OthelloState {
        uint64_t currentDiscs;
        uint64_t opponentDiscs;
        uint8_t passes;
    };
};





#endif