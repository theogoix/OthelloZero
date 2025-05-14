#include <iostream>
#include "game/othello/othello.h"

int main() {
    Othello::hello_world();
    Othello::OthelloState state = Othello::OthelloOps::initialState();
    Othello::print_othello_state(state);
    std::cout << state.currentDiscs;
    std::cout << "hello world\n";
}