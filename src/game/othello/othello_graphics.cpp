#include <iostream>
#include "othello_graphics.h"

using namespace Othello;

namespace Othello {

    void hello_world() {
        std::cout << "hello world, and goodbye";
    };

    void print_othello_state(const OthelloState& state){
        char black_symbol = 'X';
        char white_symbol = 'O';
        char empty_symbol = '.';
        bool black_turn = not (state.moveCount % 2);
        uint64_t black_discs = black_turn? state.currentDiscs : state.opponentDiscs;
        uint64_t white_discs = black_turn ? state.opponentDiscs : state.currentDiscs;
        
        std::cout << "\n#########\n";
        std::cout << (black_turn ? "Black" : "White") << "'s turn to play.\n";
        for (int irow = 0; irow < NCOL; irow++){
            for (int icol = 0; icol < NROW; icol++){
                char c;
                int isquare = icol + NCOL * irow;
                if ((black_discs >> isquare)%2){
                    
                    c = black_symbol;
                }
                else if ((white_discs >> isquare)%2) {
                    c = white_symbol;
                }
                else {c = empty_symbol;}
                std::cout << c;
            }
            std::cout<<"\n";
        }

        std::cout << "\n########\n";
    };
    

};