import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import montecarlo_cpp as mc
import time

# Simulation parameters setup
S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.3

# Domain resolution setup
sims_high = 100_000 # High-resolution batch for 2D cross-sections
sims_grid = 50_000 # Optimized batch for 3D surfaces
spots_n = 40 # Spot axis resolution
times_n = 40 # Maturity axis resolution

# Model entities initialization
process = mc.GeometricBrownianMotion(r, sigma)
payoff = mc.CallPayoff(K)
asian_opt = mc.AsianOption(payoff)
euro_opt = mc.EuropeanOption(payoff)

# Pricers instantiation
pricer_batch_high = mc.MonteCarloPricer(sims_high, 100, T)
pricer_batch_grid = mc.MonteCarloPricer(sims_grid, 50, T)

# Domain evaluation axes
spots_2d = np.linspace(70, 130, 100)
times_2d = np.full_like(spots_2d, T)
spots_3d = np.linspace(80, 120, spots_n)
times_3d = np.linspace(0.1, 1.0, times_n)
S_grid, T_grid = np.meshgrid(spots_3d, times_3d)

# C++ risk engine calculations
print("Computing risk surfaces...")
start_calc = time.time()
res_2d_euro = pricer_batch_high.calculate_all_risks_batch(spots_2d, times_2d, r, process, euro_opt)
res_2d_asian = pricer_batch_high.calculate_all_risks_batch(spots_2d, times_2d , r, process, asian_opt)
res_3d_asian = pricer_batch_grid.calculate_all_risks_batch(S_grid.flatten(), T_grid.flatten(), r, process, asian_opt)
print(f"Risk surfaces computed in {time.time()- start_calc:.2f} seconds.")

# Delta plot generation
Z_delta = res_3d_asian["deltas"].reshape(S_grid.shape)
fig1 = plt.figure(figsize=(16, 7))
ax1 = fig1.add_subplot(121)
ax1.plot(spots_2d, res_2d_euro["deltas"], 'b--', label='European')
ax1.plot(spots_2d, res_2d_asian["deltas"], 'r--', lw=2, label='Asian')
ax1.legend()
ax1.set_title("Delta 2D Comparison")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Delta")
ax2 = fig1.add_subplot(122, projection='3d')
ax2.view_init(elev=25, azim=-215)
surf1 = ax2.plot_surface(S_grid, T_grid, Z_delta, cmap='coolwarm', antialiased=True, rcount=100, ccount=100)
ax2.set_title(f"Asian Delta 3D Surface ({spots_n}x{times_n})")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Maturity (T)")
ax2.set_zlabel("Delta")
plt.savefig(f"delta_analysis_complete_{sigma}_{time.time()}.pdf")

# Vega plot generation
Z_vega = res_3d_asian["vegas"].reshape(S_grid.shape) / 100
fig2 = plt.figure(figsize=(16, 7))
ax3 = fig2.add_subplot(121)
ax3.plot(spots_2d, res_2d_euro["vegas"]/100, 'b--', label='European')
ax3.plot(spots_2d, res_2d_asian["vegas"]/100, 'r--', lw=2, label='Asian')
ax3.legend()
ax3.set_title("Vega 2D Comparison")
ax3.set_xlabel("Spot Price")
ax3.set_ylabel("Vega (1% shift)")
ax4 = fig2.add_subplot(122, projection='3d')
ax4.view_init(elev=10, azim=-235)
surf2 = ax4.plot_surface(S_grid, T_grid, Z_vega, cmap='plasma', antialiased=True, rcount=100, ccount=100)
ax4.set_title(f"Asian Vega 3D Surface ({spots_n}x{times_n})")
ax4.set_xlabel("Spot Price")
ax4.set_ylabel("Maturity (T)")
ax4.set_zlabel("Vega")
plt.savefig(f"vega_analysis_complete_{sigma}_{time.time()}.pdf")

# Gamma plot generation
Z_gamma = res_3d_asian["gammas"].reshape(S_grid.shape)
fig3 = plt.figure(figsize=(16, 7))
ax5 = fig3.add_subplot(121)
ax5.plot(spots_2d, res_2d_euro["gammas"], 'b--', label='European')
ax5.plot(spots_2d, res_2d_asian["gammas"], 'r--', lw=2, label='Asian')
ax5.legend()
ax5.set_title("Gamma 2D Comparison")
ax5.set_xlabel("Spot Price")
ax5.set_ylabel("Gamma")
ax6 = fig3.add_subplot(122, projection='3d')
ax6.view_init(elev=30, azim=-255)
surf3 = ax6.plot_surface(S_grid, T_grid, Z_gamma, cmap='inferno', antialiased=True, rcount=100, ccount=100)
ax6.set_title(f"Asian Gamma 3D Surface ({spots_n}x{times_n})")
ax6.set_xlabel("Spot Price")
ax6.set_ylabel("Maturity (T)")
ax6.set_zlabel("Gamma")
plt.savefig(f"gamma_analysis_complete_{sigma}_{time.time()}.pdf")

# Theta plot generation
Z_theta = res_3d_asian["thetas"].reshape(S_grid.shape)
fig4 = plt.figure(figsize=(16, 7))
ax7 = fig4.add_subplot(121)
ax7.plot(spots_2d, res_2d_euro["thetas"], 'b--', label='European')
ax7.plot(spots_2d, res_2d_asian["thetas"], 'r--', lw=2, label='Asian')
ax7.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax7.legend()
ax7.set_title("Theta 2D Comparison")
ax7.set_xlabel("Spot Price")
ax7.set_ylabel("Theta (Annualized)")
ax8 = fig4.add_subplot(122, projection='3d')
ax8.view_init(elev=30, azim=-135)
surf4 = ax8.plot_surface(S_grid, T_grid, Z_theta, cmap='ocean', antialiased=True, rcount=100, ccount=100)
ax8.set_title(f"Asian Theta 3D Surface ({spots_n}x{times_n})")
ax8.set_xlabel("Spot Price")
ax8.set_ylabel("Maturity (T)")
ax8.set_zlabel("Theta")

plt.savefig(f"theta_analysis_complete_{sigma}_{time.time()}.pdf")
plt.close()
