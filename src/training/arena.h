#ifndef ARENA_H
#define ARENA_H

#include <vector>
#include <string>
#include <array>
#include <memory>
#include "othello/othello.h"
#include "engine/mcts/mcts_search.h"
#include "engine/mcts/neural_evaluator.h"

namespace Training {

// Result of a single game
struct GameResult {
    int8_t outcome;  // +1 = player1 wins, -1 = player2 wins, 0 = draw
    int num_moves;
    int player1_final_discs;
    int player2_final_discs;
    std::vector<Othello::OthelloMove> move_history;
    
    GameResult() : outcome(0), num_moves(0), 
                   player1_final_discs(0), player2_final_discs(0) {}
};

// Statistics for arena matches
struct ArenaStats {
    int total_games;
    int player1_wins;
    int player2_wins;
    int draws;
    float player1_win_rate;
    float player2_win_rate;
    float draw_rate;
    float average_game_length;
    
    ArenaStats() : total_games(0), player1_wins(0), player2_wins(0), draws(0),
                   player1_win_rate(0.0f), player2_win_rate(0.0f), 
                   draw_rate(0.0f), average_game_length(0.0f) {}
    
    void update();  // Calculate rates from counts
};

// Configuration for arena matches
struct ArenaConfig {
    // Match settings
    int num_games = 100;
    bool alternate_colors = true;  // Swap who plays first each game
    
    // MCTS settings for both players
    int simulations_per_move = 800;
    float c_puct = 1.5f;
    int batch_size = 32;
    bool use_dirichlet = false;  // Usually off for evaluation
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    bool use_async = true;
    
    // Move selection
    float temperature = 0.1f;  // Low temperature for strong play
    
    // Output
    bool verbose = true;
    bool save_games = false;
    std::string output_path;
    
    ArenaConfig() = default;
};

class Arena {
public:
    Arena(
        OthelloMCTS::NeuralEvaluator& player1_evaluator,
        OthelloMCTS::NeuralEvaluator& player2_evaluator,
        const ArenaConfig& config
    );
    
    // Run full arena match
    ArenaStats run_match();
    
    // Play single game (player1 plays first if player1_first=true)
    GameResult play_game(bool player1_first = true);
    
    // Get current statistics
    const ArenaStats& get_stats() const { return stats_; }
    
    // Save results to file
    void save_results(const std::string& path) const;
    
private:
    ArenaConfig config_;
    OthelloMCTS::MCTSSearch player1_search_;
    OthelloMCTS::MCTSSearch player2_search_;
    ArenaStats stats_;
    std::vector<GameResult> game_results_;
    
    // Helper methods
    void update_stats(const GameResult& result);
    void print_game_summary(int game_number, const GameResult& result, 
                           bool player1_first) const;
    void print_final_stats() const;
    GameResult play_game_internal(
        OthelloMCTS::MCTSSearch& first_player,
        OthelloMCTS::MCTSSearch& second_player,
        bool player1_is_first
    );
};

} // namespace Training

#endif // ARENA_H