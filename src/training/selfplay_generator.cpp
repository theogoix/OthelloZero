// selfplay_generator.cpp
#include "selfplay_generator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

namespace Training {

SelfPlayGenerator::SelfPlayGenerator(
    OthelloMCTS::NeuralEvaluator& evaluator,
    const SelfPlayConfig& config
)
: config_(config)
, search_(evaluator, {
      .num_simulations = config.simulations_per_move,
      .batch_size = config.batch_size,
      .c_puct = config.c_puct,
      .use_dirichlet = config.use_dirichlet,
      .dirichlet_alpha = config.dirichlet_alpha,
      .dirichlet_epsilon = config.dirichlet_epsilon
  })
{}

void SelfPlayGenerator::generate_and_save(
    const std::string& output_path,
    int chunk_save,
    bool append
) {
    if (config_.verbose) {
        std::cout << "Starting self-play generation:\n";
        std::cout << "  Games: " << config_.num_games << "\n";
        std::cout << "  Simulations per move: " << config_.simulations_per_move << "\n";
        std::cout << "  Output: " << output_path << "\n";
        std::cout << "  Append mode: " << (append ? "yes" : "no") << "\n\n";
    }
    
    std::vector<TrainingExample> all_examples;
    all_examples.reserve(config_.num_games * 64);  // Estimate ~60 moves per game
    
    for (int game_idx = 0; game_idx < config_.num_games; game_idx++) {
        // Play one game
        std::vector<Training::TrainingExample> examples = play_game();
        all_examples.insert(all_examples.end(), examples.begin(), examples.end());
        
        total_games_++;
        total_positions_ += examples.size();
        
        if (config_.verbose && (game_idx + 1) % 10 == 0) {
            std::cout << "Progress: " << (game_idx + 1) << "/" << config_.num_games 
                     << " games, " << all_examples.size() << " positions\n";
        }

        if ((game_idx + 1) % chunk_save == 0) {
            if (config_.verbose) {
                std::cout << "Saving progress at game " << (game_idx + 1) << "...\n";
            }
            save_examples_to_file(all_examples, output_path, true);  // always append
            all_examples.clear();  // free memory
        }
    }
    
    // Save to file
    save_examples_to_file(all_examples, output_path, append);
    
    if (config_.verbose) {
        std::cout << "\nGeneration complete!\n";
        std::cout << "  Total games: " << total_games_ << "\n";
        std::cout << "  Total positions: " << total_positions_ << "\n";
        std::cout << "  Saved to: " << output_path << "\n";
    }
}

std::vector<TrainingExample> SelfPlayGenerator::play_game() {
    std::vector<TrainingExample> examples;
    examples.reserve(64);  // Typical game length
    
    
    // Initialize game
    Othello::OthelloState state = Othello::OthelloOps::initialState();
    OthelloMCTS::Node* root = nullptr;  // Start with no tree
    search_.reset();
    int move_number = 0;
    
    // Play game
    while (!Othello::OthelloOps::isTerminal(state)) {
        // Run MCTS search (with tree reuse if available)
        root = search_.search(state, root);
        
        // Extract training data
        TrainingExample example;
        example.state = state;
        example.visit_counts = search_.get_visit_counts(root);
        example.outcome = 0;  // Will be filled in after game ends
        examples.push_back(example);
        
        // Select move with temperature
        float temp = get_temperature(move_number);
        Othello::OthelloMove move = search_.select_move(root, temp);
        
        
        // Apply move
        state = Othello::OthelloOps::applyMove(state, move);
        move_number++;
        
        // Reuse tree: get child node for next iteration
        root = search_.get_child_node(root, move);
        // If child is nullptr, next search will start fresh
    }
    
    // Game finished - determine outcome
    int8_t outcome = Othello::OthelloOps::gameResult(state);
    
    
    // Update all examples with game outcome
    // Important: flip outcome for each position (alternating players)
    for (size_t i = 0; i < examples.size(); i++) {
        examples[i].outcome = outcome;
        outcome = -outcome;  // Flip for next position
    }
    
    // Print summary
    if (config_.verbose) {
        print_game_summary(total_games_ + 1, move_number, outcome);
    }
    
    return examples;
}

float SelfPlayGenerator::get_temperature(int move_number) const {
    // Use high temperature early for exploration
    // Use low temperature later to play strong moves
    if (move_number < config_.temperature_threshold_move) {
        return config_.exploration_temperature;
    } else {
        return config_.exploitation_temperature;
    }
}

void SelfPlayGenerator::save_examples_to_file(
    const std::vector<TrainingExample>& examples,
    const std::string& path,
    bool append
) {
    std::ios_base::openmode mode = std::ios::binary;
    if (append) {
        mode |= std::ios::app;
    } else {
        mode |= std::ios::trunc;
    }
    
    std::ofstream file(path, mode);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    // Write header only if creating new file
    if (!append || !std::ifstream(path).good()) {
        uint32_t magic = 0x4F544844;  // "OTHD" - Othello Training Data
        uint32_t version = 1;
        uint64_t num_examples = examples.size();
        
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&num_examples), sizeof(num_examples));
    }
    
    // Write examples
    for (const auto& ex : examples) {
        // Write state (25 bytes)
        file.write(reinterpret_cast<const char*>(&ex.state.currentDiscs), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(&ex.state.opponentDiscs), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(&ex.state.legalMoves), sizeof(uint64_t));
        uint8_t pass = ex.state.lastMoveWasPass ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&pass), sizeof(uint8_t));
        
        // Write visit counts (256 bytes)
        file.write(reinterpret_cast<const char*>(ex.visit_counts.data()), 
                   64 * sizeof(int32_t));
        
        // Write outcome (1 byte)
        file.write(reinterpret_cast<const char*>(&ex.outcome), sizeof(int8_t));
    }
    
    file.close();
}

void SelfPlayGenerator::print_game_summary(
    int game_number,
    int num_moves,
    float outcome
) const {
    const char* result_str;
    if (outcome > 0.5f) {
        result_str = "Player 1 wins";
    } else if (outcome < -0.5f) {
        result_str = "Player 2 wins";
    } else {
        result_str = "Draw";
    }
    
    std::cout << "Game " << std::setw(3) << game_number 
              << ": " << std::setw(2) << num_moves << " moves, "
              << result_str << "\n";
}

} // namespace Training