import numpy as np
from scipy.stats import norm
import time
import montecarlo_cpp as mc

# Simulation and option parameters
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.35

# Precision configuration for benchmarking
sims_euro = 10_000_000
sims_asian = 500_000
steps_euro = 1
steps_asian = 252

# Initialize stochastic process and option payoffs
process = mc.GeometricBrownianMotion(r, sigma)
payoff = mc.CallPayoff(K)
asian_opt = mc.AsianOption(payoff)
euro_opt = mc.EuropeanOption(payoff)

# Instantiate Monte Carlo pricers
pricer_euro = mc.MonteCarloPricer(sims_euro, steps_euro, T)
pricer_asian = mc.MonteCarloPricer(sims_asian, steps_asian, T)

# Analytical Black-Scholes computations
is_call = isinstance(payoff, mc.CallPayoff)
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

if is_call:
	price_bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	delta_bs = norm.cdf(d1)
	theta_bs = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
else:
	price_bs = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
	delta_bs = norm.cdf(d1) - 1.0
	theta_bs = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * 	norm.cdf(-d2)

gamma_bs = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
vega_bs = (S0 * norm.pdf(d1) * np.sqrt(T)) / 100.0

# Execution of C++ Monte Carlo risk engines
start_time = time.time()
risk_euro = pricer_euro.calculate_all_risks(S0, r, process, euro_opt)
risk_asian = pricer_asian.calculate_all_risks(S0, r, process, asian_opt)
calc_time = time.time() - start_time

# Error evaluations for European option benchmark
err_price = abs(price_bs - risk_euro.price)
err_delta = abs(delta_bs - risk_euro.delta)
err_gamma = abs(gamma_bs - risk_euro.gamma)
err_vega = abs(vega_bs - (risk_euro.vega / 100.0))
err_theta = abs(theta_bs - risk_euro.theta)

# Tabular output generation
print(f"Execution Time: {calc_time:.3f} s")
print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}\n")
print(f"{'Metric':<12} | {'BS (European)':>14} | {'MC (European)':>14} | {'Abs Error':>14} | {'MC (Asian)':>14}")
print("-" * 75)
print(f"{'Price':<12} | {price_bs:>14.6f} | {risk_euro.price:>14.6f} | {err_price:>14.8f} | {risk_asian.price:>14.6f}")
print(f"{'Delta':<12} | {delta_bs:>14.6f} | {risk_euro.delta:>14.6f} | {err_delta:>14.8f} | {risk_asian.delta:>14.6f}")
print(f"{'Gamma':<12} | {gamma_bs:>14.6f} | {risk_euro.gamma:>14.6f} | {err_gamma:>14.8f} | {risk_asian.gamma:>14.6f}")
print(f"{'Vega (1%)':<12} | {vega_bs:>14.6f} | {risk_euro.vega/100.0:>14.6f} | {err_vega:>14.8f} | {risk_asian.vega/100.0:>14.6f}")
print(f"{'Theta':<12} | {theta_bs:>14.6f} | {risk_euro.theta:>14.6f} | {err_theta:>14.8f} | {risk_asian.theta:>14.6f}")
