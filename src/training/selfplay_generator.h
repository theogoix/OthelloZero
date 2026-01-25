#ifndef SELFPLAY_GENERATOR_H
#define SELFPLAY_GENERATOR_H

#include <vector>
#include <string>
#include <array>
#include "othello/othello.h"
#include "engine/mcts/mcts_search.h"
#include "engine/mcts/neural_evaluator.h"

namespace Training {

// Single training example from one position
struct TrainingExample {
    Othello::OthelloState state;
    std::array<float, 64> policy;  // Raw visit counts from MCTS
    int8_t outcome;  // -1 = loss, 0 = draw, 1 = win (current player's perspective)
    float q_value;


    
    TrainingExample() : outcome(0) {
        policy.fill(0);
    }
};


struct PositionData {
    Othello::OthelloState state;           // Full state (25 bytes)
    std::array<int32_t, 64> visit_counts;  // MCTS visits (256 bytes)
    std::array<float, 64> q_values;        // Q-value for each action (256 bytes)
    Othello::OthelloMove move_played;      // Move that was actually selected (1 byte)
    float temperature;                      // Temperature used (4 bytes)
    uint32_t thinking_time_ms;             // Time spent (4 bytes)
};

struct GameMetadata {
    uint32_t game_id;
    int8_t outcome;                        // Final result: +1, 0, -1
    uint8_t player1_disc_count;            // Final disc counts
    uint8_t player2_disc_count;
    std::string timestamp;
    
    // All positions in chronological order
    std::vector<PositionData> positions;
    
    // Derived properties
    uint16_t num_moves() const { return positions.size(); }
};



// Configuration for self-play generation
struct SelfPlayConfig {
    // Game generation
    int num_games = 100;
    int simulations_per_move = 800;
    
    // Temperature schedule for move selection
    int temperature_threshold_move = 30;  // Use exploration temp until this move
    float exploration_temperature = 1.0f;  // High temp = more exploration
    float exploitation_temperature = 0.1f; // Low temp = play best move
    
    // MCTS configuration
    float c_puct = 1.5f;
    int batch_size = 32;
    bool use_dirichlet = true;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    bool use_async = true;
    
    // Output
    bool verbose = true;
};

class SelfPlayGenerator {
public:
    SelfPlayGenerator(
        OthelloMCTS::NeuralEvaluator& evaluator,
        const SelfPlayConfig& config
    );
    
    // Generate multiple games and save to file
    void generate_and_save(
        const std::string& output_path,
        int chunck_save = 50,
        bool append = true
    );

    
    // Generate single game (returns all positions from the game)
    std::vector<TrainingExample> play_game();
    
    // Get statistics
    int get_total_games_generated() const { return total_games_; }
    int get_total_positions_generated() const { return total_positions_; }
    
private:
    SelfPlayConfig config_;
    OthelloMCTS::MCTSSearch search_;
    
    // Statistics
    int total_games_ = 0;
    int total_positions_ = 0;
    
    // Helper methods
    float get_temperature(int move_number) const;
    
    void save_examples_to_file(
        const std::vector<TrainingExample>& examples,
        const std::string& path,
        bool append
    );
    
    void print_game_summary(
        int game_number,
        int num_moves,
        float outcome
    ) const;
};

} // namespace Training

#endif // SELFPLAY_GENERATOR_H