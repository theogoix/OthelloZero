#include "training/arena.h"

int main(int argc, char** argv) {
    // Load your models

    std::string base_path = argv[1];
    std::string challenger_path = argv[2];
    OthelloMCTS::NeuralEvaluator base_model(base_path);
    OthelloMCTS::NeuralEvaluator challenger(challenger_path);
    
    // Configure arena
    Training::ArenaConfig config;


    config.num_games = 100;
    config.simulations_per_move = 800;
    config.alternate_colors = true;
    config.temperature = 0.1f;  // Low temp = strong play
    config.use_async = true;
    config.verbose = true;
    config.save_games = true;
    config.output_path = "arena_results.csv";


    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--games" && i + 1 < argc) {
            config.num_games = std::atoi(argv[++i]);
        } else if (arg == "--simulations" && i + 1 < argc) {
            config.simulations_per_move = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            config.temperature = std::atof(argv[++i]);
        } else if (arg == "--c-puct" && i + 1 < argc) {
            config.c_puct = std::atof(argv[++i]);
        } else if (arg == "--no-async") {
            config.use_async = false;
        } else if (arg == "--output-csv") {
            config.output_path = argv[++i];
        } else if (arg == "--quiet") {
            config.verbose = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    
    // Create arena
    Training::Arena arena(base_model, challenger, config);
    
    // Run match
    Training::ArenaStats stats = arena.run_match();
    
    // Check if challenger is better
    if (stats.player1_win_rate > 0.55f) {
        std::cout << "Challenger wins! Promoting to new base model.\n";
    }
    
    return 0;
}