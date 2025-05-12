#ifndef OTHELLO_BOARD_H
#define OTHELLO_BOARD_H

#include <cstdint>


const int N_ROWS = 8;
const int N_COLS = 8;
using Bitboard = uint64_t;

struct OthelloBoard {
    Bitboard black_discs;
    Bitboard white_discs;
};


const OthelloBoard StartingBoard = {
    0x0000001008000000,
    0x0000000810000000
};

struct OthelloCoordinate {
    int i_col;
    int i_row;
};

struct OthelloState {};

int coordinate_to_index(OthelloCoordinate coord);

void print_bitboard(OthelloBoard bitboard);

#endif
