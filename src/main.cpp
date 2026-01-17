#include <iostream>
#include <chrono>
#include "engine/mcts/mcts_search.h"
#include "engine/mcts/neural_evaluator.h"
#include "othello/othello.h"

using namespace OthelloMCTS;

int main(int argc, char** argv) {
    std::cout << "Othello MCTS with Batched Neural Network Evaluation\n";
    std::cout << "===================================================\n\n";
    
    // Configuration
    MCTSConfig config;
    config.num_simulations = 800;
    config.batch_size = 32;
    config.c_puct = 1.5f;
    
    // Choose evaluator
    std::unique_ptr<NeuralEvaluator> nn_eval;
    RandomEvaluator random_eval;
    
    
    if (argc > 1) {
        // Try to load neural network from file
        std::string model_path = argv[1];
        std::cout << "Loading neural network from: " << model_path << "\n";
        
        try {
            nn_eval = std::make_unique<NeuralEvaluator>(model_path, true);
            if (nn_eval->is_available()) {
                std::cout << "Using neural network evaluator\n\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to load NN, falling back to random: " 
                      << e.what() << "\n\n";
        }

    }
    
    // Create MCTS search
    MCTSSearch search(*nn_eval, config);
    Node* root = nullptr;

    // Play a few moves
    auto state = Othello::OthelloOps::initialState();
    bool black_to_play = true;
    

    for (int move_num = 0; move_num < 64; move_num++) {
        std::cout << "\n=== Move " << (move_num + 1) << " ===\n";
        std::cout << "Current player: " 
                  << (state.currentDiscs == 0 ? "Black (X)" : "White (O)")
                  << "\n\n";
        

        Othello::print_othello_state(state, black_to_play);
        
        // Check if terminal
        if (Othello::OthelloOps::isTerminal(state)) {
            std::cout << "Game over!\n";
            float result = Othello::OthelloOps::gameResult(state);
            if (result > 0) {
                std::cout << "Winner: " 
                          << (state.currentDiscs == 0 ? "Black" : "White") 
                          << "\n";
            } else if (result < 0) {
                std::cout << "Winner: " 
                          << (state.currentDiscs == 0 ? "White" : "Black") 
                          << "\n";
            } else {
                std::cout << "Draw!\n";
            }
            break;
        }
        
        // Run MCTS search
        auto start = std::chrono::high_resolution_clock::now();
        
        root = search.search(state,root);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start);
        
        std::cout << "Search time: " << duration.count() << " ms\n";
        
        Othello::OthelloMove best_move = search.select_move(root, 1.0f);

        if (best_move == Othello::OthelloMove::PASS) {
            std::cout << "No legal moves available\n";
            break;
        }
        
        // Show move in algebraic notation
        int row = best_move / 8;
        int col = best_move % 8;
        char col_letter = 'a' + col;
        std::cout << "Selected move: " << col_letter << (row + 1) << "\n";
        
        // Apply move
        state = Othello::OthelloOps::applyMove(state, best_move);
        
        // Reset search for next move (clears tree)
        root = search.get_child_node(root, best_move);
        black_to_play = !black_to_play;
    }

    std::cout << "\nDemo complete!\n";
    
    return 0;
}

// Example random evaluator wrapper for demo
// In real code, you'd properly handle the polymorphism