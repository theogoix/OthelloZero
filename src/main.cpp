#include <iostream>
#include "othello/othello.h"

int main() {
    using Ops = Othello::OthelloOps;
    Othello::OthelloState state = Othello::OthelloOps::initialState();
    std::vector<Othello::OthelloMove> moves = Ops::generateMoves(state);
    std::cout << moves.size();
    for (int i = 0; i < moves.size() ; i++){
        Othello::OthelloMove& mv = moves.at(i);
        std::cout << "applying movie " << mv << "\n";
        Othello::OthelloState new_state = Ops::applyMove(state, mv);
        Othello::print_othello_state(new_state);
    }
    Othello::print_othello_state(state);
    std::cout << "hello world\n";
}