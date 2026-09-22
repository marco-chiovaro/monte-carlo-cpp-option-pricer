import numpy as np
import montecarlo_cpp

spot = 100.0
strike = 100.0
rate = 0.05
volatility = 0.80
maturity = 1.0

simulations = 10000000
time_steps = 1

pricer = montecarlo_cpp.MonteCarloPricer(simulations, time_steps, maturity)
gbm = montecarlo_cpp.GeometricBrownianMotion(rate, volatility)

raw_paths = pricer.generate_paths(spot, gbm)
paths = np.array(raw_paths).reshape(simulations, time_steps + 1)
terminal_spots = paths[:, -1]

discount_factor = np.exp(-rate * maturity)

call_payoffs = np.maximum(terminal_spots - strike, 0.0) * discount_factor
put_payoffs = np.maximum(strike - terminal_spots, 0.0) * discount_factor

call_premium = np.mean(call_payoffs)
put_premium = np.mean(put_payoffs)

losses_dict = {
	"Long Call": call_premium - call_payoffs,
	"Short Call": call_payoffs - call_premium,
	"Long Put": put_premium - put_payoffs,
	"Short Put": put_payoffs - put_premium,
}

print(f"European Call Premium: {call_premium:.4f}")
print(f"European Put Premium:  {put_premium:.4f}\n")

print("-" * 66)
print(f"{'Position':<12} {'95% VaR':<10} {'95% ES':<10} {'99% VaR':<10} {'99% ES':<10} {'Max Loss':<10}")
print("-" * 66)

for position_name, losses in losses_dict.items():
	var_95 = np.percentile(losses, 95)
	es_95 = np.mean(losses[losses >= var_95])

	var_99 = np.percentile(losses, 99)
	es_99 = np.mean(losses[losses >= var_99])

	max_loss = np.max(losses)

	print(f"{position_name:<12} {var_95:<10.4f} {es_95:<10.4f} {var_99:<10.4f} {es_99:<10.4f} {max_loss:<10.4f}")

print("-" * 66)