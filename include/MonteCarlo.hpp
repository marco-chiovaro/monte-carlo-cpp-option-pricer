	
#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP
	
#include "Process.hpp"
#include "Option.hpp" 
#include <vector>
#include <cstdint>
	
// Result structure for a single configuration (AoS - Array of Structures)
struct RiskMetrics {
	double price;
	double delta;
	double gamma;
	double vega;
	double theta;
};
	
// Result structure for a single configuration (AoS - Array of Structures)
struct RiskSurface {
	std::vector<double> prices;
	std::vector<double> deltas;
	std::vector<double> gammas;
	std::vector<double> vegas;
	std::vector<double> thetas;
};
	
class MonteCarloPricer {
	private:
		double maturity;
		long num_simulations;
		int num_steps;

	public:
		
		MonteCarloPricer(long sims, int steps, double T);
		
		[[nodiscard]] inline long get_simulations() const { return num_simulations; }
		[[nodiscard]] inline int get_steps() const { return num_steps; }
		[[nodiscard]] inline double get_maturity() const { return maturity; }
		
		
		double price(double initial_spot, double discount_rate, const StochasticProcess& process, const Option& option) const;
		
		std::vector<double> generate_paths(double initial_spot,const StochasticProcess& process) const;


		// Helper function for seeding
		double price_with_seed(double initial_spot, double r, const StochasticProcess& process, const Option& option, uint64_t seed) const;
		
		std::vector<double> price_batch( const std::vector<double>& spots, const std::vector<double>& maturities, double r, 	const StochasticProcess& process, const Option& option) const;
		
		// Calculates everything for a single Spot and Time
		RiskMetrics calculate_all_risks(double initial_spot, double r, const GeometricBrownianMotion& process, const Option& option) const;
		
		// Calculates everything for a grid of Spots and Times
		RiskSurface calculate_all_risks_batch(const std::vector<double>& spot_grid,	const std::vector<double>& time_grid, double r, const GeometricBrownianMotion& process, const Option& option) const;
};
	
#endif // MONTECARLO_HPP
