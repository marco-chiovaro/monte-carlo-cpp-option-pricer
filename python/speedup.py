import os
import subprocess
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
n_sims = [10000, 50000, 100000, 500000, 1000000, 1500000]

NUM_THREADS_PARALLEL = 8

def run_benchmark():
	import montecarlo_cpp as mc
	process = mc.GeometricBrownianMotion(r, sigma)
	payoff = mc.CallPayoff(K)
	option = mc.EuropeanOption(payoff)
	execution_times = []
	for n in n_sims:
		pricer = mc.MonteCarloPricer(n, 100, T)
		start = time.perf_counter()
		pricer.price(S0, r, process, option)
		elapsed = time.perf_counter() - start
		execution_times.append(elapsed)
	return execution_times

def main_worker():
	times = run_benchmark()
	print(",".join(map(str, times)))

def launch_test(num_threads):
	env = os.environ.copy()
	env["OMP_NUM_THREADS"] = str(num_threads)
	result = subprocess.run(
		[sys.executable, sys.argv[0], "worker"],
		capture_output=True,
		text=True,
		env=env,
	)
	
	return np.array([float(x) for x in result.stdout.strip().split(",")])

if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == "worker":
		main_worker()
		sys.exit()

	print("Running single-thread benchmark...")
	times_single = launch_test(1)

	print(f"Running OpenMP benchmark ({NUM_THREADS_PARALLEL} threads)...")
	times_parallel = launch_test(NUM_THREADS_PARALLEL)

	plt.figure(figsize=(10, 6))
	plt.plot(
		n_sims,
		times_single,
		"ro-",
		linewidth=2,
		label="Single Thread (Sequential)",
	)
	
	plt.plot(
		n_sims,
		times_parallel,
		"go-",
		linewidth=2,
		label=f"OpenMP ({NUM_THREADS_PARALLEL} Threads)",
	)

	speedup = np.mean(times_single / times_parallel)
	plt.xlabel("Number of Simulations ($N$)")
	plt.ylabel("Execution Time (seconds)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	plt.savefig(
		"openmp_performance_comparison.png", dpi=300, bbox_inches="tight"
	)
	
	plt.close()
	print(f"Plot saved successfully. Average speedup: {speedup:.2f}x")
