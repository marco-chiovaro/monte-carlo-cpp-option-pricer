import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
	
import montecarlo_cpp
	
	
def black_scholes_call(S, K, T, r, sigma):
	d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	
	
def plot_convergence(S0, K, T, r, sigma):
	process = montecarlo_cpp.GeometricBrownianMotion(r, sigma)
	payoff = montecarlo_cpp.CallPayoff(K)
	option = montecarlo_cpp.EuropeanOption(payoff)
	
	n_steps = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
	results = []
	
	print("Starting convergence analysis...")
	for n in n_steps:
		pricer = montecarlo_cpp.MonteCarloPricer(n, 100, T)
		price = pricer.price(S0, r, process, option)
		results.append(price)
		
		print(f"N = {n:7} | Price = {price:.4f}")
	
	plt.figure(figsize=(10, 5))
	plt.plot(n_steps, results, marker='o', linestyle='-', color='teal', label='Monte Carlo Estimate')
	plt.axhline(y=10.4506, color='red', linestyle='--', label='Black-Scholes Target') 
	
	plt.xscale('log')
	plt.title("Price Convergence vs Number of Simulations ($N$)")
	plt.xlabel("Simulations (N) - Log Scale")
	plt.ylabel("Estimated Price")
	plt.legend()
	plt.grid(True, which="both", alpha=0.3)
	plt.savefig("convergence_plot.png")
	plt.close()
	
	
def save_montecarlo_3d_png_v2(pricer, spot, process, K, filename="montecarlo_surface.png"):
	matrix = pricer.generate_paths(spot, process)
	T = pricer.maturity
	
	n_plot = min(150, matrix.shape[0])
	data = matrix[:n_plot, :]
	n_sims, n_steps = data.shape
	
	time_axis = np.linspace(0, T, n_steps)
	sim_axis = np.arange(n_sims)
	Time, Sims = np.meshgrid(time_axis, sim_axis)
	
	fig = plt.figure(figsize=(18, 12))
	ax = fig.add_subplot(111, projection='3d')
	
	surf = ax.plot_surface(Time, Sims, data, cmap='ocean',
	edgecolor='black', linewidth=0.05, alpha=0.85, antialiased=True, rstride=1, cstride=1) 
	
	strike_plane = np.full_like(data, K)
	ax.plot_surface(Time, Sims, strike_plane, color='red', alpha=0.15, shade=False)      
	
	ax.contourf(Time, Sims, data, zdir='x', offset=T, cmap='magma', alpha=0.3)
	ax.contourf(Time, Sims, data, zdir='y', offset=n_sims, cmap='magma', alpha=0.1)
	
	ax.set_xlabel(r'Time to Maturity ($t \in [0, T]$)', fontsize=14, labelpad=15)
	ax.set_ylabel(r'Simulation Index ($N$)', fontsize=14, labelpad=15)
	ax.set_zlabel(r'Underlying Price ($S_t$)', fontsize=14, labelpad=15)
	
	title = (r'Stochastic Outcome Domain' + '\n' +
	r'Monte Carlo Path Surface (GBM Parallel Engine)')
	ax.set_title(title, fontsize=20, pad=30)
	
	ax.set_xlim(0, T)
	ax.set_ylim(0, n_sims)
	
	ax.view_init(elev=30, azim=-125)
	
	cbar = fig.colorbar(surf, shrink=0.5, aspect=15, pad=0.1)
	cbar.set_label(r'Asset Value ($S_t$)', rotation=270, labelpad=20)
	
	plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
	plt.close()
	
	
def plot_option_value_surface(pricer, process, option, S0, K, T, r, filename="surface.pdf"):
	spots = np.linspace(S0 * 0.7, S0 * 1.3, 40) 
	times = np.linspace(0.01, T, 40)
	
	S, Time = np.meshgrid(spots, times)
	Z = np.zeros_like(S)
	
	for i in range(len(times)):
		current_time = float(times[i])
		temp_pricer = montecarlo_cpp.MonteCarloPricer(10000, 100, current_time)
	
		spots_row = spots.tolist()
		times_row = [current_time] * len(spots)
	
		Z[i, :] = temp_pricer.price_batch(spots_row, times_row, float(r), process, option)
	
	fig = plt.figure(figsize=(16, 10))
	ax = fig.add_subplot(111, projection='3d')
	
	surf = ax.plot_surface(S, Time, Z, cmap='viridis', 
	edgecolor='black', linewidth=0.1, alpha=0.9)
	
	ax.set_xlabel(r'Underlying Price ($S$)', fontsize=12)
	ax.set_ylabel(r'Time to Maturity ($\tau$)', fontsize=12)
	ax.set_zlabel(r'Option Price ($V$)', fontsize=12)
	ax.set_title(r'Option Value Surface - Monte Carlo Batch Engine', fontsize=16, pad=20)
	
	ax.invert_yaxis() 
	ax.view_init(elev=20, azim=-135)
	fig.colorbar(surf, shrink=0.5, aspect=10)
	
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	
	
def main():
	S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.3, 1.0
	sims_price = 100000
	sims_plot = 100
	sims_hist = 10000
	steps_europe = 1
	steps_asian = 252
	
	gbm = montecarlo_cpp.GeometricBrownianMotion(r, sigma)
	base_call_payoff = montecarlo_cpp.CallPayoff(K)
	
	european_option = montecarlo_cpp.EuropeanOption(base_call_payoff)
	asian_option = montecarlo_cpp.AsianOption(base_call_payoff)
	
	pricer_value_europe = montecarlo_cpp.MonteCarloPricer(sims_price, steps_europe, T)
	pricer_value_asian = montecarlo_cpp.MonteCarloPricer(sims_price, steps_asian, T)
	
	bs_price = black_scholes_call(S0, K, T, r, sigma)
	
	start_euro = time.time()
	p_euro = pricer_value_europe.price(S0, r, gbm, european_option)
	t_euro = time.time() - start_euro
	
	start_asian = time.time()
	p_asian = pricer_value_asian.price(S0, r, gbm, asian_option) 
	t_asian = time.time() - start_asian
	
	error_emp = abs(p_euro - bs_price)
	
	pricer_sample = montecarlo_cpp.MonteCarloPricer(10000, 1, T)
	sample_paths = pricer_sample.generate_paths(S0, gbm)
	
	sample_finals = np.array(sample_paths)[:, -1]
	sample_payoffs = np.maximum(sample_finals - K, 0) * np.exp(-r * T)
	sigma_payoff = np.std(sample_payoffs, ddof=1)
	
	error_theo = sigma_payoff / np.sqrt(sims_price)
	
	print("-" * 50)
	print(f"PRICING RESULTS (Execution Time: {t_euro + t_asian:.4f}s)")
	print("-" * 50)
	print(f"Analytical BS Price:  {bs_price:.6f}")
	print(f"European MC Price:   {p_euro:.6f}  (Time: {t_euro:.4f}s)")
	print(f"Asian MC Price:      {p_asian:.6f}  (Time: {t_asian:.4f}s)")
	print("-" * 50)
	print(f"Empirical Error:     {error_emp:.6f}")
	print(f"Theoretical Error:   {error_theo:.6f} (Sigma / sqrt(N))")
	print(f"Payoff Std Deviation:{sigma_payoff:.4f}")
	print("-" * 50)
	
	# Value at Risk (VaR) Analysis
	pricer_var = montecarlo_cpp.MonteCarloPricer(1000000, 1, T)
	paths_var = pricer_var.generate_paths(S0, gbm)
	final_prices_var = np.array(paths_var)[:, -1]
	
	payoffs_var = np.maximum(final_prices_var - K, 0) * np.exp(-r * T)
	pnl = payoffs_var - bs_price
	
	percentile_5 = np.percentile(pnl, 5)
	percentile_1 = np.percentile(pnl, 1)
	
	var_95 = abs(percentile_5) if percentile_5 < 0 else 0.0
	var_99 = abs(percentile_1) if percentile_1 < 0 else 0.0
	
	print("\nVALUE AT RISK (VaR) ANALYSIS")
	print(f"Option Premium Cost: {bs_price:.4f}")
	print(f"95% VaR:              {var_95:.4f}")
	print(f"99% VaR:              {var_99:.4f}")
	print("-" * 50)
	
	# Figure Generation
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	folder = 'simulation_plots'
	if not os.path.exists(folder): 
		os.makedirs(folder)
	
	# Path Trajectories Plot
	pricer_plot = montecarlo_cpp.MonteCarloPricer(sims_plot, steps_asian, T)
	paths = pricer_plot.generate_paths(S0, gbm)
	time_axis = np.linspace(0, T, steps_asian + 1)
	
	plt.figure(figsize=(10, 6))
	for path in paths: 
		plt.plot(time_axis, path, lw=1, alpha=0.4)
	plt.title(f'Monte Carlo Trajectories - N={sims_plot} - Time: {t_euro+t_asian:.2f}s')
	plt.xlabel('Time to Maturity')
	plt.ylabel('Asset Price')
	plt.savefig(f"{folder}/trajectories_{timestamp}.pdf")
	plt.close()
	
	# Terminal Distribution Histogram
	pricer_hist = montecarlo_cpp.MonteCarloPricer(sims_hist, steps_asian, T)
	paths_hist = pricer_hist.generate_paths(S0, gbm)
	final_prices = np.array(paths_hist)[:, -1]
	
	plt.figure(figsize=(10, 6))
	plt.hist(final_prices, bins=50, color='skyblue', edgecolor='black')
	plt.title(f'Terminal Asset Price Distribution (MC Error: {error_emp:.5f})')
	plt.xlabel('Price at Maturity')
	plt.ylabel('Frequency')
	plt.savefig(f"{folder}/distribution_histogram_{timestamp}.png")
	plt.close()
	
	# 3D Path Surface Rendering
	steps_3d = 252 
	sims_3d = 150
	pricer_3d = montecarlo_cpp.MonteCarloPricer(sims_3d, steps_3d, T)
	filename_3d = f"{folder}/surface_3d_{timestamp}.pdf"
	save_montecarlo_3d_png_v2(pricer_3d, S0, gbm, K, filename=filename_3d)
	
	# Option Value Surfaces
	plot_option_value_surface(
	pricer_value_europe,
	gbm,
	european_option,
	S0, K, T, r,
	filename=f"{folder}/value_surface_european_{timestamp}.pdf"
	)
	
	plot_option_value_surface(
	pricer_value_asian,
	gbm,
	asian_option,
	S0, K, T, r,
	filename=f"{folder}/value_surface_asian_{timestamp}.pdf"
	)
	
	
if __name__ == "__main__":
	main()
