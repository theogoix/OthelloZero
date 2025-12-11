#include "othello_ops.h"
#include <iostream>

using namespace Othello;

using Bitboard = uint64_t;

constexpr Bitboard COL_H = 0x0101010101010101ULL;
constexpr Bitboard COL_A = 0x8080808080808080ULL;
constexpr Bitboard ROW_1 = 0xFF00000000000000ULL;
constexpr Bitboard ROW_8 = 0x00000000000000FFULL;

constexpr Bitboard NOT_COL_H = ~COL_H;
constexpr Bitboard NOT_COL_A = ~COL_A;
constexpr Bitboard EMPTY = 0x0000000000000000ULL;

Bitboard square_to_bb(OthelloMove sq){return 1ULL << sq;};

void print_bb(Bitboard bb){
    for (int i = 0; i < 8; i++){
        for (int j = 0; j < 8; j++){
            if ((bb >> (8 * i + j))%2){
                std::cout << 'P';
            }
            else {std::cout << '.';}
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

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

Bitboard generateMovesBb(const OthelloState& state){
    Bitboard cur = state.currentDiscs;
    Bitboard opp = state.opponentDiscs;
    Bitboard emp = ~(cur | opp);
    print_bb(0x0000FFFF00001111);
    print_bb(cur);
    print_bb(opp);

    Bitboard candidate_moves = EMPTY;
    
    for (int i = 0; i < 8; i++){
        std::cout << "Going for loop " << i << "\n";
        DirectionMask direction_mask = direction_mask_lookup[i];
        Direction dir = direction_mask.dir;
        Bitboard mask = direction_mask.mask; 
        Bitboard gen = cur;
        Bitboard pro = (opp | emp) & ~(emp << dir);
        Bitboard dir_fill = fill(gen, pro, dir, mask);
        Bitboard cand_mov_dir = dir_fill & emp;
        print_bb(cand_mov_dir);
        candidate_moves |= cand_mov_dir;
    };
    print_bb(candidate_moves);

    return candidate_moves;
};

namespace Othello {

    std::vector<OthelloMove> OthelloOps::generateMoves(const OthelloState& state){
        std::vector<OthelloMove> move_list = {};

        Bitboard candidate_moves = generateMovesBb(state);
        if (candidate_moves == 0){
            move_list.push_back(PASS);
        }
        else {
            while (candidate_moves){
                OthelloMove mv = static_cast<OthelloMove> (__builtin_clzll(candidate_moves));
                candidate_moves &= ~(-(candidate_moves));
            }
        }
        return move_list;
    };

    OthelloState OthelloOps::applyMove(const OthelloState& state, const OthelloMove& move){
        Bitboard cur = state.currentDiscs;
        Bitboard opp = state.opponentDiscs;
        uint8_t passes = state.passes;
        if (move == PASS){
            passes++;
        }
        else {
            passes = 0;
            Bitboard gen = square_to_bb(move);

                for (int i = 0; i < 8; i++){
                    DirectionMask direction_mask = direction_mask_lookup[i];
                    Direction dir = direction_mask.dir;
                    Bitboard mask = direction_mask.mask; 
                    Bitboard pro = (opp | cur) & ~(cur << dir);
                    Bitboard dir_fill = fill(gen, pro, dir, mask);
                    Bitboard target = dir_fill & cur & (~gen);
                    if (target){
                        Bitboard flipped = dir_fill & opp;
                        opp &= ~flipped;
                        cur |= flipped;
                    }
                };
            }
        return {
            opp,
            cur,
            passes,
        };
    };
    
    bool OthelloOps::isTerminal(const OthelloState& state){
        return state.passes > 1;
    };

    OthelloState OthelloOps::initialState(){
        return {
        0x0000001008000000,
        0x0000000810000000,
        0,
        };
    };
};