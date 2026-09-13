#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP

#include "Process.hpp"
#include "Option.hpp" // Aggiungiamo l'inclusione della nuova classe Option
#include <vector>
#include <cstdint>

// Struttura per il risultato di una singola configurazione (AoS - Array of Structures)
struct RiskMetrics {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
};

// Struttura per le superfici 3D ottimizzata per Python/NumPy (SoA - Structure of Arrays)
struct RiskSurface {
    std::vector<double> prices;
    std::vector<double> deltas;
    std::vector<double> gammas;
    std::vector<double> vegas;
    std::vector<double> thetas;
};

class MonteCarloPricer {
private:
    //double maturity;
    double maturity;
    long num_simulations;
    int num_steps;
    

public:
    
    MonteCarloPricer(long sims, int steps, double T);

    // --- GETTERS (Zero-cost abstraction) ---
    [[nodiscard]] inline long get_simulations() const { return num_simulations; }
    [[nodiscard]] inline int get_steps() const { return num_steps; }
    [[nodiscard]] inline double get_maturity() const { return maturity; }
    // Con [[nodiscard]] se non venisse utilizzato il valore comparirà un Warning durante la compilazione (Modern C++)

    // ---> IL NUOVO METODO UNIVERSALE <---
    // Invece di passare il Payoff, passiamo la classe base Option!
    double price(double initial_spot, double discount_rate, 
                 const StochasticProcess& process, 
                 const Option& option) const;
    
    double price_with_seed(double initial_spot, double r, const StochasticProcess& process, const Option& option, uint64_t seed) const;
    
    std::vector<double> generate_paths(double initial_spot, const StochasticProcess& process) const;

    std::vector<double> price_batch(
        const std::vector<double>& spots, 
        const std::vector<double>& maturities,
        double r, 
        const StochasticProcess& process, 
        const Option& option) const;

    // 1. Calcola tutto per un singolo Spot e Tempo
    RiskMetrics calculate_all_risks(double initial_spot, double r, 
                                    const GeometricBrownianMotion& process, 
                                    const Option& option) const;

    // 2. Calcola tutto per una griglia di Spot e Tempi (Userà OpenMP)
    RiskSurface calculate_all_risks_batch(const std::vector<double>& spot_grid, 
                                          const std::vector<double>& time_grid, 
                                          double r, 
                                          const GeometricBrownianMotion& process, 
                                          const Option& option) const;
};

#endif // MONTECARLO_HPP