#include "othello_ops.h"

using namespace Othello;

using Bitboard = uint64_t;

constexpr Bitboard COL_H = 0x0101010101010101ULL;
constexpr Bitboard COL_A = 0x8080808080808080ULL;
constexpr Bitboard ROW_1 = 0xFF00000000000000ULL;
constexpr Bitboard ROW_8 = 0x00000000000000FFULL;

constexpr Bitboard NOT_COL_H = ~COL_H;
constexpr Bitboard NOT_COL_A = ~COL_A;
constexpr Bitboard EMPTY = 0x0000000000000000ULL;


enum Direction {
    EAST = 1,
    SOUTH = 8,
    WEST = -EAST,
    NORTH = -SOUTH,
    SOUTHEAST = SOUTH + EAST,
    SOUTHWEST = SOUTH + WEST,
    NORTHWEST = NORTH + WEST,
    NORTHEAST = NORTH + EAST,

};

struct DirectionMask {
    Direction dir;
    Bitboard mask;
};

const DirectionMask direction_mask_lookup[8] = {
    {EAST, NOT_COL_H},
    {SOUTH, EMPTY},
    {WEST, NOT_COL_A},
    {NORTH, EMPTY},
    {SOUTHEAST, NOT_COL_H},
    {SOUTHWEST, NOT_COL_A},
    {NORTHWEST, NOT_COL_A},
    {NORTHEAST, NOT_COL_H},
};

Bitboard fill(Bitboard gen, Bitboard pro, Direction dir, Bitboard mask){
    pro &= mask;
    gen |= pro & (gen << dir);
    pro &= (pro << dir);
    gen |= pro & (gen << 2 * dir);
    pro &= (pro << 2 * dir);
    gen |= pro & (gen << 4 * dir);
    return gen;
};

Bitboard southFill(Bitboard g, Bitboard p){
    g |= p & (g << SOUTH);
    p &= (p << SOUTH);
    g |= p & (g << 2 * SOUTH);
    p &= (p << 2*SOUTH);
    g |= p & (g << 4 * SOUTH);
    
    return g;
    
};



namespace Othello {

    std::vector<OthelloMove> OthelloOps::generateMoves(const OthelloState& state){
        std::vector<OthelloMove> move_list = {};

        Bitboard cur = state.currentDiscs;
        Bitboard opp = state.opponentDiscs;
        Bitboard emp = ~(cur | opp);

        Bitboard candidate_moves = EMPTY;
        
        for (int i = 0; i < 8; i++){
            DirectionMask direction_mask = direction_mask_lookup[i];
            Direction dir = direction_mask.dir;
            Bitboard mask = direction_mask.mask; 
            Bitboard gen = cur;
            Bitboard pro = (opp | emp) & ~(emp << dir);
            Bitboard dir_fill = fill(gen, pro, dir, mask);
            Bitboard cand_mov_dir = dir_fill & emp;
            candidate_moves |= cand_mov_dir;
        };

        return move_list;
    };

    OthelloState OthelloOps::applyMove(const OthelloState& state, const OthelloMove& move){
        return state;
    };
    
    bool OthelloOps::isTerminal(const OthelloState& state){
        return state.moveCount < 60;
    };

    OthelloState OthelloOps::initialState(){
        return {
        0x0000001008000000,
        0x0000000810000000,
        0,
        };
    };
};