#ifndef OTHELLOSTATE_H
#define OTHELLOSTATE_H

#include<cstdint>

namespace Othello{
    
    const int NCOL = 8;
    const int NROW = 8;
    

    struct OthelloState {
        uint64_t currentDiscs;
        uint64_t opponentDiscs;
        uint64_t legalMoves;
        bool lastMoveWasPass;
        
        bool operator==(const OthelloState& other) const {
        return currentDiscs == other.currentDiscs &&
               opponentDiscs == other.opponentDiscs &&
               lastMoveWasPass == other.lastMoveWasPass;
        }
    };
    
    enum Result {
        Loss = -1,
        Draw = 0,
        Win = 1,
    };




};





#endif