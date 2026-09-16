import time
import numpy as np
from scipy.stats import norm
import montecarlo_cpp
	
	
def black_scholes_call(S, K, T, r, sigma):
	d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	
	
def main():
	
	S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
	steps_european = 1
	steps_asian = 252
	
	total_sims = 10_000_000
	num_batches = 50
	sims_per_batch = total_sims // num_batches
	
	print(f"\nRunning {total_sims:,} total simulations...")
	print(f"Split into {num_batches} batches of {sims_per_batch:,} simulations each.")
	
	gbm = montecarlo_cpp.GeometricBrownianMotion(r, sigma)
	base_call_payoff = montecarlo_cpp.CallPayoff(K)
	
	european_option = montecarlo_cpp.EuropeanOption(base_call_payoff)
	asian_option = montecarlo_cpp.AsianOption(base_call_payoff)
	
	pricer_european = montecarlo_cpp.MonteCarloPricer(sims_per_batch, steps_european, T)
	pricer_asian = montecarlo_cpp.MonteCarloPricer(sims_per_batch, steps_asian, T)
	
	bs_price = black_scholes_call(S0, K, T, r, sigma)

	# European Option Benchmark
	euro_batch_prices = []
	start_euro = time.time()
	for _ in range(num_batches):
		euro_batch_prices.append(pricer_european.price(S0, r, gbm, european_option))
	end_euro = time.time()
	t_euro = end_euro - start_euro
	
	euro_array = np.array(euro_batch_prices)
	p_euro = np.mean(euro_array)
	se_euro = np.std(euro_array, ddof=1) / np.sqrt(num_batches)
	error_emp = abs(p_euro - bs_price)
	
	# Asian Option Benchmark
	asian_batch_prices = []
	start_asian = time.time()
	for _ in range(num_batches):
		asian_batch_prices.append(pricer_asian.price(S0, r, gbm, asian_option))
	end_asian = time.time()
	t_asian = end_asian - start_asian
	
	asian_array = np.array(asian_batch_prices)
	p_asian = np.mean(asian_array)
	se_asian = np.std(asian_array, ddof=1) / np.sqrt(num_batches)
	
	print("\n" + "=" * 60)
	print(" PRICING RESULTS AND STATISTICAL ANALYSIS")
	print("=" * 60)
	print(f"Black-Scholes Target: {bs_price:.6f} €\n")
	
	print("--- EUROPEAN OPTION ---")
	print(f"MC Mean Price:    {p_euro:.6f} €")
	print(f"Standard Error:   {se_euro:.6f} €")
	print(f"95% Confidence:   [{p_euro - 1.96*se_euro:.6f} € , {p_euro + 1.96*se_euro:.6f} €]")
	print(f"Empirical Error:  {error_emp:.6f} €")
	print(f"Execution Time:   {t_euro:.4f} seconds\n")
	
	print("--- ASIAN OPTION ---")
	print(f"MC Mean Price:    {p_asian:.6f} €")
	print(f"Standard Error:   {se_asian:.6f} €")
	print(f"95% Confidence:   [{p_asian - 1.96*se_asian:.6f} € , {p_asian + 1.96*se_asian:.6f} €]")
	print(f"Execution Time:   {t_asian:.4f} seconds")
	print("=" * 60)
	
	
if __name__ == "__main__":
	main()
