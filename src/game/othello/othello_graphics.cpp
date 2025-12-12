#include <iostream>
#include "othello_graphics.h"
#include <cctype>

using namespace Othello;

namespace Othello {

    void hello_world() {
        std::cout << "hello world, and goodbye";
    };

    void print_othello_state(const OthelloState& state, bool black_turn){
        char black_symbol = 'X';
        char white_symbol = 'O';
        char empty_symbol = '.';
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

    OthelloState from_fen(const std::string& fen){
        OthelloState state = {
            0x0000000000000000ULL,
            0x0000000000000000ULL,
            0,
        };
        int row = 0;
        int col = 0;

        for (char c : fen) {
            if (c == '/') { row++; col = 0; continue; }
            if (std::isdigit(c)) { col += c - '0'; continue; }

            int pos = row * 8 + col;
            if (c == 'X') state.currentDiscs |= (1ULL << pos);
            else if (c == 'O') state.opponentDiscs |= (1ULL << pos);

            col++;
        }
        return state;
    }
    

};