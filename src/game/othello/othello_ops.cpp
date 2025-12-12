#include "othello_ops.h"
#include <iostream>


using namespace Othello;

using Bitboard = uint64_t;

constexpr Bitboard COL_A = 0x0101010101010101ULL;
constexpr Bitboard COL_H = 0x8080808080808080ULL;
constexpr Bitboard ROW_1 = 0x00000000000000FFULL;
constexpr Bitboard ROW_8 = 0xFF00000000000000ULL;


constexpr Bitboard NOT_COL_A = ~COL_A;
constexpr Bitboard NOT_COL_H = ~COL_H;
constexpr Bitboard EMPTY = 0x0000000000000000ULL;
constexpr Bitboard FULL = ~EMPTY;


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


enum DirectionId {
    EAST_ID,
    SOUTH_ID,
    WEST_ID,
    NORTH_ID,
    SOUTHEAST_ID,
    SOUTHWEST_ID,
    NORTHWEST_ID,
    NORTHEAST_ID,
};

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

constexpr DirectionMask direction_mask_lookup[8] = {
    {EAST, NOT_COL_A},
    {SOUTH, FULL},
    {WEST, NOT_COL_H},
    {NORTH, FULL},
    {SOUTHEAST, NOT_COL_A},
    {SOUTHWEST, NOT_COL_H},
    {NORTHWEST, NOT_COL_H},
    {NORTHEAST, NOT_COL_A},
};

template<int dir>
constexpr Bitboard shift(Bitboard bb){
    if constexpr (dir > 0){
        return bb << dir;
    }
    else if constexpr (dir < 0){
        return bb >> -dir;
    }
    else {
        return bb;
    }
}

template<int dir_id>
constexpr Bitboard shift_id(Bitboard bb){
    constexpr Direction dir = direction_mask_lookup[dir_id].dir;
    return shift<dir>(bb); 
}

template<int direction_id>
constexpr Bitboard fill(Bitboard gen, Bitboard pro){
    constexpr DirectionMask dirmask = direction_mask_lookup[direction_id];
    constexpr Direction dir = dirmask.dir;
    constexpr Bitboard mask = dirmask.mask;
    pro &= mask;
    gen |= pro & shift<dir>(gen);
    pro &= shift<dir>(pro);
    gen |= pro & shift<2 * dir>(gen);
    pro &= shift<2 * dir>(pro);
    gen |= pro & shift<4 * dir>(gen);
    return gen;
};


Bitboard generateMovesBb(const OthelloState& state){
    Bitboard cur = state.currentDiscs;
    Bitboard opp = state.opponentDiscs;
    Bitboard emp = ~(cur | opp);

    Bitboard candidate_moves = EMPTY;

    #define GENERATE_MOVES_DIR(id) \
        do { \
            Bitboard gen = cur;\
            Bitboard pro = opp | ( emp & shift_id<id>(opp));\
            Bitboard dir_fill = fill<id>(gen, pro);\
            Bitboard cand_mov_dir = dir_fill & emp;\
            candidate_moves |= cand_mov_dir;\
        } while(0);

    GENERATE_MOVES_DIR(0);
    GENERATE_MOVES_DIR(1);
    GENERATE_MOVES_DIR(2);
    GENERATE_MOVES_DIR(3);
    GENERATE_MOVES_DIR(4);
    GENERATE_MOVES_DIR(5);
    GENERATE_MOVES_DIR(6);
    GENERATE_MOVES_DIR(7);

    #undef GENERATE_MOVES_DIR


    //print_bb(candidate_moves);

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
                OthelloMove mv = static_cast<OthelloMove>(__builtin_ctzll(candidate_moves));
                move_list.push_back(mv);
                candidate_moves &= candidate_moves - 1;
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
            
            #define FLIP_DIR(id) \
                do {\
                    constexpr DirectionMask direction_mask = direction_mask_lookup[id]; \
                    constexpr Direction dir = direction_mask.dir; \
                    Bitboard pro = (opp | cur) & ~(shift<dir>(cur)); \
                    Bitboard dir_fill = fill<id>(gen, pro); \
                    Bitboard target = dir_fill & cur & (~gen); \
                    if (target){ \
                        Bitboard flipped = dir_fill & opp; \
                        opp &= ~flipped; \
                        cur |= flipped; \
                    }\
                } while(0);
            FLIP_DIR(0);
            FLIP_DIR(1);
            FLIP_DIR(2);
            FLIP_DIR(3);
            FLIP_DIR(4);
            FLIP_DIR(5);
            FLIP_DIR(6);
            FLIP_DIR(7);

            #undef FLIPDIR
            cur |= gen;

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
        0x0000000810000000,
        0x0000001008000000,
        0,
        };
    };
};