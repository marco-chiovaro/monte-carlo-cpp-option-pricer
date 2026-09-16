#include "../include/MonteCarlo.hpp"
#include "../include/Process.hpp"
#include "../include/Payoff.hpp"
#include "../include/Xoshiro256.hpp"

#include <cmath>
#include <vector>
#include <random>
#include <atomic>
#include <omp.h>
#include <cstdint>
#include <algorithm>


// STOCHASTIC PROCESSES IMPLEMENTATION (Geometric Brownian Motion)

GeometricBrownianMotion::GeometricBrownianMotion(double r, double vol) :
	drift(r), volatility(vol) {}

double GeometricBrownianMotion::evolve(double current_spot, double dt, double random_gaussian) const {
	double exponent = (drift - 0.5 * volatility * volatility) * dt + (volatility * std::sqrt(dt) * random_gaussian);
	return current_spot * std::exp(exponent);
}

double GeometricBrownianMotion::get_drift() const {
	return drift;
}

double GeometricBrownianMotion::get_volatility() const {
	return volatility;
}


// PRICING ENGINE IMPLEMENTATION (MonteCarloPricer)

MonteCarloPricer::MonteCarloPricer(long sims, int steps, double T): 
	maturity(T), num_simulations(sims), num_steps(steps) {}

double MonteCarloPricer::price(double initial_spot, double discount_rate,
const StochasticProcess& process,
const Option& option) const {
	double dt = maturity / num_steps;
	double sum_payoffs = 0.0;
	long iterations = num_simulations / 2;
	
	// Base master seed computed only once
	static uint64_t master_seed = []() {
		std::random_device rd;
		return (static_cast<uint64_t>(rd()) << 32) | rd();
	}();
	
	// Counter surviving across function calls
	static std::atomic<uint64_t> call_offset{0};
	
	// Advance the offset based on how many threads will be used in this call.
	// fetch_add returns the current value and safely increments it.
	int max_threads = omp_get_max_threads();
	uint64_t current_offset = call_offset.fetch_add(max_threads);
	
	#pragma omp parallel reduction(+:sum_payoffs)
	{
		int thread_id = omp_get_thread_num();
		// Thread-local initialization: master_seed + call_offset + thread_id
		// Guarantees that no thread across any batch will ever share the same seed.
		Xoshiro256StarStar rng(master_seed + current_offset + thread_id);
		std::normal_distribution<double> norm(0.0, 1.0);
		std::vector<double> path1(num_steps + 1);
		std::vector<double> path2(num_steps + 1);
		
		#pragma omp for
		for (long i = 0; i < iterations; ++i) {
			path1[0] = initial_spot;
			path2[0] = initial_spot;
			double current_spot1 = initial_spot;
			double current_spot2 = initial_spot;
			
			for (int j = 1; j <= num_steps; ++j) {
				double z = norm(rng);
				current_spot1 = process.evolve(current_spot1, dt, z);
				current_spot2 = process.evolve(current_spot2, dt, -z);
				path1[j] = current_spot1;
				path2[j] = current_spot2;
			}
			
			double p1 = option.get_payoff(path1);
			double p2 = option.get_payoff(path2);
			sum_payoffs += (p1 + p2) / 2.0;
		}
	}
	return (sum_payoffs / iterations) * std::exp(-discount_rate * maturity);
}


// PRICE WITH SEED IMPLEMENTATION

double MonteCarloPricer::price_with_seed(double spot, double rate,
const StochasticProcess& process,
const Option& option,
uint64_t base_seed) const {
	double dt = maturity / num_steps;
	double sum_payoffs = 0.0;
	long iterations = num_simulations / 2; // Halved for antithetic variates
	
	// Parallelization via OpenMP
	#pragma omp parallel reduction(+:sum_payoffs)
	{
		int thread_id = omp_get_thread_num();
		
		// Thread-private vector allocation (prevents race conditions)
		Xoshiro256StarStar rng(base_seed + thread_id);
		std::normal_distribution<double> norm(0.0, 1.0);
		
		// Thread-private vector allocation (prevents race conditions)
		std::vector<double> path1(num_steps + 1);
		std::vector<double> path2(num_steps + 1);
		
		// schedule(static) ensures identical paths for a given seed
		#pragma omp for schedule(static)
		for (long i = 0; i < iterations; ++i) {
			path1[0] = spot;
			path2[0] = spot;
			double current_spot1 = spot;
			double current_spot2 = spot;
			
			for (int j = 1; j <= num_steps; ++j) {
				double z = norm(rng);
				// Antithetic variates: z and -z
				current_spot1 = process.evolve(current_spot1, dt, z);
				current_spot2 = process.evolve(current_spot2, dt, -z);
				path1[j] = current_spot1;
				path2[j] = current_spot2;
			}
			
			// Compute average payoff passing the generated paths
			sum_payoffs += (option.get_payoff(path1) + option.get_payoff(path2)) / 2.0;
		}
	}
	return (sum_payoffs / iterations) * std::exp(-rate * maturity);
}

// Batch Price Calculation (Option Value Surface)
std::vector<double> MonteCarloPricer::price_batch(const std::vector<double>& spots, 
const std::vector<double>& maturities, double r, const StochasticProcess& process, 
const Option& option) const {
	size_t n = spots.size();
	std::vector<double> results(n);
	int num_cores = omp_get_max_threads();
	
	#pragma omp parallel for num_threads(num_cores) schedule(static)
	for (int i = 0; i < static_cast<int>(n); ++i) {
		// Deterministic seed for reproducibility
		unsigned int fixed_seed = 12345;
		// Pure pricing call without finite-difference bumping
		results[i] = price_with_seed(spots[i], r, process, option, fixed_seed);
	}
	return results;
}

RiskMetrics MonteCarloPricer::calculate_all_risks(double initial_spot, double r, 
const GeometricBrownianMotion& process, const Option& option) const {
	double dt = maturity / num_steps;
	
	// Bump sizes for finite differences
	double h_S = initial_spot * 0.001;
	double h_V = 0.001;
	double h_T = 1.0 / 365.0;
	
	bool calc_theta = (maturity > h_T);
	double dt_theta = calc_theta ? (maturity - h_T) / num_steps : 0.0;
	
	double current_vol = process.get_volatility();
	GeometricBrownianMotion proc_up_V(r, current_vol + h_V);
	GeometricBrownianMotion proc_down_V(r, current_vol - h_V);
	
	// Accumulators for mean payoffs
	double sum_base = 0.0, sum_up_S = 0.0, sum_down_S = 0.0;
	double sum_up_V = 0.0, sum_down_V = 0.0, sum_theta = 0.0;
	
	// Dual Vector allocation
	std::vector<double> path_base1(num_steps + 1), path_base2(num_steps + 1);
	std::vector<double> path_up_S1(num_steps + 1), path_up_S2(num_steps + 1);
	std::vector<double> path_down_S1(num_steps + 1), path_down_S2(num_steps + 1);
	std::vector<double> path_up_V1(num_steps + 1), path_up_V2(num_steps + 1);
	std::vector<double> path_down_V1(num_steps + 1), path_down_V2(num_steps + 1);
	std::vector<double> path_theta1(num_steps + 1), path_theta2(num_steps + 1);
	
	Xoshiro256StarStar rng(12345);
	std::normal_distribution<double> norm(0.0, 1.0);
	
	// Halved iterations
	long iterations = num_simulations / 2;
	
	for (long i = 0; i < iterations; ++i) {
		// Spot initialization at t=0 for both paths
		path_base1[0]   = initial_spot;
		path_base2[0]   = initial_spot;
		path_up_S1[0]   = initial_spot + h_S;
		path_up_S2[0]   = initial_spot + h_S;
		path_down_S1[0] = initial_spot - h_S;
		path_down_S2[0] = initial_spot - h_S;
		path_up_V1[0]   = initial_spot;
		path_up_V2[0]   = initial_spot;
		path_down_V1[0] = initial_spot;
		path_down_V2[0] = initial_spot;
		path_theta1[0]  = initial_spot;
		path_theta2[0]  = initial_spot;
		
		// Current spot variables
		double S_base1 = path_base1[0], S_base2 = path_base2[0];
		double S_up1 = path_up_S1[0], S_up2 = path_up_S2[0];
		double S_down1 = path_down_S1[0], S_down2 = path_down_S2[0];
		double V_up1 = path_up_V1[0], V_up2 = path_up_V2[0];
		double V_down1 = path_down_V1[0], V_down2 = path_down_V2[0];
		double S_theta1 = path_theta1[0], S_theta2 = path_theta2[0];
		
		// Path generation
		for (int j = 1; j <= num_steps; ++j) {
			double z = norm(rng);
			
			// Path 1 evolution (positive z)
			S_base1 = process.evolve(S_base1, dt, z);
			S_up1   = process.evolve(S_up1, dt, z);
			S_down1 = process.evolve(S_down1, dt, z);
			V_up1   = proc_up_V.evolve(V_up1, dt, z);
			V_down1 = proc_down_V.evolve(V_down1, dt, z);
			
			// Path 2 evolution (negative z)
			S_base2 = process.evolve(S_base2, dt, -z);
			S_up2   = process.evolve(S_up2, dt, -z);
			S_down2 = process.evolve(S_down2, dt, -z);
			V_up2   = proc_up_V.evolve(V_up2, dt, -z);
			V_down2 = proc_down_V.evolve(V_down2, dt, -z);
			
			// Store values in vectors
			path_base1[j]   = S_base1;
			path_base2[j]   = S_base2;
			path_up_S1[j]   = S_up1;
			path_up_S2[j]   = S_up2;
			path_down_S1[j] = S_down1;
			path_down_S2[j] = S_down2;
			path_up_V1[j]   = V_up1;
			path_up_V2[j]   = V_up2;
			path_down_V1[j] = V_down1;
			path_down_V2[j] = V_down2;
			
			if (calc_theta) {
				S_theta1 = process.evolve(S_theta1, dt_theta, z);
				S_theta2 = process.evolve(S_theta2, dt_theta, -z);
				path_theta1[j] = S_theta1;
				path_theta2[j] = S_theta2;
			}
		}
		
		// Compute averaged payoffs between standard and antithetic paths
		sum_base   += (option.get_payoff(path_base1)   + option.get_payoff(path_base2))   / 2.0;
		sum_up_S   += (option.get_payoff(path_up_S1)   + option.get_payoff(path_up_S2))   / 2.0;
		sum_down_S += (option.get_payoff(path_down_S1) + option.get_payoff(path_down_S2)) / 2.0;
		sum_up_V   += (option.get_payoff(path_up_V1)   + option.get_payoff(path_up_V2))   / 2.0;
		sum_down_V += (option.get_payoff(path_down_V1) + option.get_payoff(path_down_V2)) / 2.0;
		
		if (calc_theta) {
			sum_theta += (option.get_payoff(path_theta1) + option.get_payoff(path_theta2)) / 2.0;
		}
	}
	
	// Compute discounted averages
	double discount = std::exp(-r * maturity);
	double avg_base   = (sum_base / iterations) * discount;
	double avg_up_S   = (sum_up_S / iterations) * discount;
	double avg_down_S = (sum_down_S / iterations) * discount;
	double avg_up_V   = (sum_up_V / iterations) * discount;
	double avg_down_V = (sum_down_V / iterations) * discount;
	
	RiskMetrics result;
	result.price = avg_base;
	result.delta = (avg_up_S - avg_down_S) / (2.0 * h_S);
	result.gamma = (avg_up_S - 2.0 * avg_base + avg_down_S) / (h_S * h_S);
	result.vega  = (avg_up_V - avg_down_V) / (2.0 * h_V);
	
	if (calc_theta) {
		double discount_theta = std::exp(-r * (maturity - h_T));
		double avg_theta = (sum_theta / iterations) * discount_theta;
		result.theta = (avg_theta - avg_base) / h_T;
	} else {
		result.theta = 0.0;
	}
	
	return result;
}

RiskSurface MonteCarloPricer::calculate_all_risks_batch(
const std::vector<double>& spot_grid, const std::vector<double>& time_grid,
double r, const GeometricBrownianMotion& process,const Option& option) const {
	
	size_t n = spot_grid.size();
	RiskSurface surface;
	surface.prices.resize(n);
	surface.deltas.resize(n);
	surface.gammas.resize(n);
	surface.vegas.resize(n);
	surface.thetas.resize(n);
	
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < n; ++i) {
		// Create a local pricer instance with 'maturity' set to time_grid[i]
		MonteCarloPricer local_pricer(num_simulations, num_steps, time_grid[i]);
		RiskMetrics point_metrics = local_pricer.calculate_all_risks(spot_grid[i], r, process, option);
		
		surface.prices[i] = point_metrics.price;
		surface.deltas[i] = point_metrics.delta;
		surface.gammas[i] = point_metrics.gamma;
		surface.vegas[i]  = point_metrics.vega;
		surface.thetas[i] = point_metrics.theta;
	}
	
	return surface;
}

std::vector<double> MonteCarloPricer::generate_paths(double initial_spot,
const StochasticProcess& process) const {
	double dt = maturity / num_steps;
	std::vector<double> paths(num_simulations * (num_steps + 1));
	
	// Static initialization
	static uint64_t master_seed = []() {
		std::random_device rd;
		return (static_cast<uint64_t>(rd()) << 32) | rd();
	}();
	
	// Atomic counter to track function calls
	static std::atomic<uint64_t> call_offset{0};
	
	// Shift the offset
	int max_threads = omp_get_max_threads();
	uint64_t current_offset = call_offset.fetch_add(max_threads);
	
	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		// Mathematically guaranteed unique seed
		Xoshiro256StarStar rng(master_seed + current_offset + thread_id);
		std::normal_distribution<double> norm(0.0, 1.0);
		
		#pragma omp for schedule(static)
		for (long i = 0; i < num_simulations; ++i) {
			double current_spot = initial_spot;
			long row_start = i * (num_steps + 1);
			paths[row_start] = current_spot;
			
			for (int j = 1; j <= num_steps; ++j) {
				double z = norm(rng);
				current_spot = process.evolve(current_spot, dt, z);
				paths[row_start + j] = current_spot;
			}
		}
	}
	
	return paths;
}
