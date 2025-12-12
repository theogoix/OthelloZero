#include <iostream>
#include "game/othello/othello.h"

int main() {
    using Ops = Othello::OthelloOps;
    Othello::OthelloState state = Othello::OthelloOps::initialState();
    std::cout << Ops::generateMoves(state).size() << "\n";
    Othello::print_othello_state(state);
    std::cout << "hello world\n";
}