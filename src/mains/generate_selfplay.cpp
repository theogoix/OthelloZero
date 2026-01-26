#include <iostream>
#include <exception>
#include <string>
#include "training/selfplay_generator.h"
#include "engine/mcts/neural_evaluator.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.pt> <output.bin> [options]\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  model.pt     Path to TorchScript model\n";
    std::cout << "  output.bin   Path to output training data file\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  --games N              Number of games to generate (default: 100)\n";
    std::cout << "  --simulations N        MCTS simulations per move (default: 800)\n";
    std::cout << "  --temp-threshold N     Move number to switch temperature (default: 30)\n";
    std::cout << "  --explore-temp T       Exploration temperature (default: 1.0)\n";
    std::cout << "  --exploit-temp T       Exploitation temperature (default: 0.1)\n";
    std::cout << "  --c-puct C             CPUCT constant";
    std::cout << "  --no-append            Overwrite output file instead of appending\n";
    std::cout << "  --no-async             Use non-async version of search";
    std::cout << "  --quiet                Suppress progress output\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Required arguments
    std::string model_path = argv[1];
    std::string output_path = argv[2];
    
    // Parse optional arguments
    Training::SelfPlayConfig config;
    bool append_mode = true;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--games" && i + 1 < argc) {
            config.num_games = std::atoi(argv[++i]);
        } else if (arg == "--simulations" && i + 1 < argc) {
            config.simulations_per_move = std::atoi(argv[++i]);
        } else if (arg == "--temp-threshold" && i + 1 < argc) {
            config.temperature_threshold_move = std::atoi(argv[++i]);
        } else if (arg == "--explore-temp" && i + 1 < argc) {
            config.exploration_temperature = std::atof(argv[++i]);
        } else if (arg == "--exploit-temp" && i + 1 < argc) {
            config.exploitation_temperature = std::atof(argv[++i]);
        } else if (arg == "--c-puct" && i + 1 < argc) {
            config.c_puct = std::atof(argv[++i]);
        } else if (arg == "--no-append") {
            append_mode = false;
        } else if (arg == "--no-async") {
            config.use_async = false;
        } else if (arg == "--quiet") {
            config.verbose = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        // Load neural network
        std::cout << "Loading model from: " << model_path << "\n";
        OthelloMCTS::NeuralEvaluator evaluator(model_path, true);
        
        if (!evaluator.is_available()) {
            std::cerr << "Error: Failed to load neural network\n";
            return 1;
        }
        
        std::cout << "Model loaded successfully\n\n";
        
        // Create generator and run
        Training::SelfPlayGenerator generator(evaluator, config);
        generator.generate_and_save(output_path, 50, append_mode);
        
        std::cout << "\n✓ Success!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}