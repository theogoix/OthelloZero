#include<iostream>
#include "board.h"


int coordinate_to_index(OthelloCoordinate coord){
    return coord.i_col + N_COLS * coord.i_row;
};

void print_bitboard(OthelloBoard board){
    std::cout << "\n#######\n";
    for (int i_row = 0; i_row < N_ROWS; i_row++){
        for (int i_col = 0; i_col < N_COLS; i_col++){
            int offset = coordinate_to_index({i_col,i_row});
            if ((board.black_discs >> offset) & 1)  {
                std::cout << "x";
            }
            else if ((board.white_discs >> offset) & 1) {
                std::cout << "o";
            }
            else {
                std::cout << ".";
            }

        };
        std::cout << "\n";
    };
};