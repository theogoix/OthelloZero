#include "arena.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

namespace Training {

void ArenaStats::update() {
    if (total_games == 0) {
        player1_win_rate = 0.0f;
        player2_win_rate = 0.0f;
        draw_rate = 0.0f;
        return;
    }
    
    player1_win_rate = static_cast<float>(player1_wins) / total_games;
    player2_win_rate = static_cast<float>(player2_wins) / total_games;
    draw_rate = static_cast<float>(draws) / total_games;
}

Arena::Arena(
    OthelloMCTS::NeuralEvaluator& player1_evaluator,
    OthelloMCTS::NeuralEvaluator& player2_evaluator,
    const ArenaConfig& config
)
: config_(config)
, player1_search_(player1_evaluator, {
      .num_simulations = config.simulations_per_move,
      .batch_size = config.batch_size,
      .c_puct = config.c_puct,
      .use_dirichlet = config.use_dirichlet,
      .dirichlet_alpha = config.dirichlet_alpha,
      .dirichlet_epsilon = config.dirichlet_epsilon
  })
, player2_search_(player2_evaluator, {
      .num_simulations = config.simulations_per_move,
      .batch_size = config.batch_size,
      .c_puct = config.c_puct,
      .use_dirichlet = config.use_dirichlet,
      .dirichlet_alpha = config.dirichlet_alpha,
      .dirichlet_epsilon = config.dirichlet_epsilon
  })
{
    if (config_.save_games) {
        game_results_.reserve(config_.num_games);
    }
}

ArenaStats Arena::run_match() {
    if (config_.verbose) {
        std::cout << "\n=== Arena Match Started ===\n";
        std::cout << "Games to play: " << config_.num_games << "\n";
        std::cout << "Simulations per move: " << config_.simulations_per_move << "\n";
        std::cout << "Alternate colors: " << (config_.alternate_colors ? "Yes" : "No") << "\n";
        std::cout << "Temperature: " << config_.temperature << "\n";
        std::cout << "Using " << (config_.use_async ? "async" : "non-async") << " search\n";
        std::cout << "===========================\n\n";
    }
    
    stats_ = ArenaStats();
    if (config_.save_games) {
        game_results_.clear();
    }
    
    for (int game_idx = 0; game_idx < config_.num_games; game_idx++) {
        bool player1_first = true;
        
        // Alternate who plays first
        if (config_.alternate_colors) {
            player1_first = (game_idx % 2 == 0);
        }
        
        GameResult result = play_game(player1_first);
        
        update_stats(result);
        
        if (config_.save_games) {
            game_results_.push_back(result);
        }
        
        if (config_.verbose) {
            print_game_summary(game_idx + 1, result, player1_first);
            
            // Print interim stats every 10 games
            if ((game_idx + 1) % 10 == 0) {
                std::cout << "\n--- Progress: " << (game_idx + 1) << "/" 
                         << config_.num_games << " ---\n";
                std::cout << "Player 1: " << stats_.player1_wins << " wins ("
                         << std::fixed << std::setprecision(1)
                         << (stats_.player1_win_rate * 100) << "%)\n";
                std::cout << "Player 2: " << stats_.player2_wins << " wins ("
                         << (stats_.player2_win_rate * 100) << "%)\n";
                std::cout << "Draws: " << stats_.draws << " ("
                         << (stats_.draw_rate * 100) << "%)\n\n";
            }
        }
    }
    
    if (config_.verbose) {
        print_final_stats();
    }
    
    if (config_.save_games && !config_.output_path.empty()) {
        save_results(config_.output_path);
    }
    
    return stats_;
}

GameResult Arena::play_game(bool player1_first) {
    if (player1_first) {
        return play_game_internal(player1_search_, player2_search_, true);
    } else {
        return play_game_internal(player2_search_, player1_search_, false);
    }
}

GameResult Arena::play_game_internal(
    OthelloMCTS::MCTSSearch& first_player,
    OthelloMCTS::MCTSSearch& second_player,
    bool player1_is_first
) {
    GameResult result;
    
    // Initialize game
    Othello::OthelloState state = Othello::OthelloOps::initialState();
    OthelloMCTS::Node* first_root = nullptr;
    OthelloMCTS::Node* second_root = nullptr;
    
    first_player.reset();
    second_player.reset();
    
    int move_number = 0;
    bool first_player_turn = true;
    
    // Play game
    while (!Othello::OthelloOps::isTerminal(state)) {
        OthelloMCTS::MCTSSearch& current_search = first_player_turn ? 
                                                   first_player : second_player;
        OthelloMCTS::Node*& current_root = first_player_turn ? 
                                            first_root : second_root;
        
        // Run MCTS search
        current_root = config_.use_async ? 
                       current_search.search_async(state, current_root) :
                       current_search.search(state, current_root);
        
        // Select move with low temperature for strong play
        Othello::OthelloMove move = current_search.select_move(
            current_root, 
            config_.temperature
        );
        
        // Record move
        result.move_history.push_back(move);
        
        // Apply move
        state = Othello::OthelloOps::applyMove(state, move);
        move_number++;
        
        // Reuse tree for current player
        current_root = current_search.get_child_node(current_root, move);
        
        // For the opponent, we need to find the corresponding child
        // in their tree (if they have one)
        OthelloMCTS::Node*& opponent_root = first_player_turn ? 
                                             second_root : first_root;
        if (opponent_root != nullptr) {
            // Try to reuse opponent's tree by finding the child node
            // corresponding to the move just played
            opponent_root = (first_player_turn ? second_player : first_player)
                           .get_child_node(opponent_root, move);
        }
        
        // Switch turns
        first_player_turn = !first_player_turn;
    }
    
    // Game finished - determine outcome
    int8_t game_outcome = Othello::OthelloOps::gameResult(state);


    // Since it returns according to player 1 point of view
    // The game result is accurate if it was both from its point
    // of view and if he played first, or if none of it occurred
    if (first_player_turn == player1_is_first){
        result.outcome = game_outcome;
    }
    else {
        result.outcome = -game_outcome;
    }
    
    
    result.num_moves = move_number;
    
    // Count final discs (from player1's perspective)
    if (first_player_turn == player1_is_first) {
        result.player1_final_discs = __builtin_popcountll(state.opponentDiscs);
        result.player2_final_discs = __builtin_popcountll(state.currentDiscs);
    } else {
        result.player1_final_discs = __builtin_popcountll(state.currentDiscs);
        result.player2_final_discs = __builtin_popcountll(state.opponentDiscs);
    }
    
    return result;
}

void Arena::update_stats(const GameResult& result) {
    stats_.total_games++;
    
    if (result.outcome > 0) {
        stats_.player1_wins++;
    } else if (result.outcome < 0) {
        stats_.player2_wins++;
    } else {
        stats_.draws++;
    }
    
    stats_.average_game_length = 
        (stats_.average_game_length * (stats_.total_games - 1) + result.num_moves) 
        / stats_.total_games;
    
    stats_.update();
}

void Arena::print_game_summary(
    int game_number, 
    const GameResult& result,
    bool player1_first
) const {
    std::cout << "Game " << std::setw(3) << game_number << ": ";
    
    // Show who played first
    std::cout << (player1_first ? "P1-P2" : "P2-P1") << " | ";
    
    // Show result
    if (result.outcome > 0) {
        std::cout << "Player 1 wins";
    } else if (result.outcome < 0) {
        std::cout << "Player 2 wins";
    } else {
        std::cout << "Draw";
    }
    
    // Show score
    std::cout << " (" << result.player1_final_discs << "-" 
              << result.player2_final_discs << ")";
    
    // Show number of moves
    std::cout << " in " << result.num_moves << " moves\n";
}

void Arena::print_final_stats() const {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║        ARENA MATCH RESULTS            ║\n";
    std::cout << "╠════════════════════════════════════════╣\n";
    std::cout << "║ Total games:     " << std::setw(4) << stats_.total_games 
              << "                  ║\n";
    std::cout << "║                                        ║\n";
    std::cout << "║ Player 1 wins:   " << std::setw(4) << stats_.player1_wins 
              << " (" << std::fixed << std::setprecision(1) << std::setw(5)
              << (stats_.player1_win_rate * 100) << "%)      ║\n";
    std::cout << "║ Player 2 wins:   " << std::setw(4) << stats_.player2_wins 
              << " (" << std::setw(5) << (stats_.player2_win_rate * 100) 
              << "%)      ║\n";
    std::cout << "║ Draws:           " << std::setw(4) << stats_.draws 
              << " (" << std::setw(5) << (stats_.draw_rate * 100) 
              << "%)      ║\n";
    std::cout << "║                                        ║\n";
    std::cout << "║ Avg game length: " << std::fixed << std::setprecision(1)
              << std::setw(5) << stats_.average_game_length 
              << " moves            ║\n";
    
    // Calculate win rate difference and confidence
    float win_rate_diff = stats_.player1_win_rate - stats_.player2_win_rate;
    float std_error = std::sqrt(
        stats_.player1_win_rate * (1 - stats_.player1_win_rate) / stats_.total_games
    );
    float z_score = std::abs(win_rate_diff) / (std_error + 1e-6f);
    
    std::cout << "║                                        ║\n";
    std::cout << "║ Win rate diff:   " << std::showpos << std::setw(6)
              << (win_rate_diff * 100) << std::noshowpos 
              << "%             ║\n";
    
    if (z_score > 1.96f) {  // 95% confidence
        std::cout << "║ Significance:    SIGNIFICANT (95%+)   ║\n";
    } else if (z_score > 1.645f) {  // 90% confidence
        std::cout << "║ Significance:    LIKELY (90%+)        ║\n";
    } else {
        std::cout << "║ Significance:    INCONCLUSIVE         ║\n";
    }
    
    std::cout << "╚════════════════════════════════════════╝\n\n";
}

void Arena::save_results(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return;
    }
    
    // Write header
    file << "game_number,player1_first,outcome,player1_discs,player2_discs,num_moves\n";
    
    // Write each game result
    for (size_t i = 0; i < game_results_.size(); i++) {
        const auto& result = game_results_[i];
        bool player1_first = !config_.alternate_colors || (i % 2 == 0);
        
        file << (i + 1) << ","
             << (player1_first ? 1 : 0) << ","
             << static_cast<int>(result.outcome) << ","
             << result.player1_final_discs << ","
             << result.player2_final_discs << ","
             << result.num_moves << "\n";
    }
    
    file.close();
    
    if (config_.verbose) {
        std::cout << "Results saved to: " << path << "\n";
    }
}

} // namespace Training