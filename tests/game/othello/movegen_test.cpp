#include "doctest.h"
#include "game/othello/othello.h"

using namespace Othello;


TEST_CASE("Testing the fen generator"){
    OthelloState initial_state = OthelloOps::initialState();

    OthelloState initial_state_fen = from_fen("8/8/8/3OX3/3XO3/8/8/8");
    CHECK(initial_state == initial_state_fen);
}


TEST_CASE("on the upper top corner"){
    OthelloState state = {
        0x0000000000000101,
        0x0000000000000002,
        0,
    };
    CHECK(OthelloOps::generateMoves(state).size() == 1);
    CHECK(OthelloOps::generateMoves(state).at(0)== C1);
}




TEST_CASE("Testing the move generation manually"){


    struct MoveTestCase {
        std::string fen;
        std::vector<OthelloMove> expected;
    };

    MoveTestCase tests[] = {
        {"8/8/8/3OX3/3XO3/8/8/8", {D3,C4,F5,E6}},
        {"///3OX3/3XO3///", {D3,C4,F5,E6}},
        {"XO6///////", {C1}},
        {"OX6/XXX5//////", {PASS}},
        {"2OX4/2XXO3/2XOO3/3XO3/4OX2/5O2/8/8", {B1, F2, F3, F4, D5, F7, G7}},
        // more cases
    };

    for (auto &t : tests) {
    SUBCASE(t.fen.c_str()) {
        OthelloState state = from_fen(t.fen);
        print_othello_state(state);
        auto moves = OthelloOps::generateMoves(state);
        CAPTURE(t.fen);
        CHECK(moves == t.expected);
    }
    }


}

TEST_CASE("Check win condition"){
    OthelloState state = from_fen("XXXXXXXX/OOOOOOOO/XXXXXXXX/OOOOOOOO/XXXXXXXX/OOOOOOOO/XXXXXXXX/OOOOOOOO");
    Result result = OthelloOps::gameResult(state);
    CHECK(result == Result::Draw);
};

TEST_CASE("Initial position move generation") {

    OthelloState state = OthelloOps::initialState();


    std::vector<OthelloMove> moves = OthelloOps::generateMoves(state);

    // Expected moves (example indices)
    std::vector<OthelloMove> expected = {D3, C4, F5, E6};


    // Sort if necessary
    CHECK(moves == expected);
}

