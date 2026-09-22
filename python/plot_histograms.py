import matplotlib.pyplot as plt
import numpy as np
import montecarlo_cpp

# Data generetaion via C++ monte carlo engine
simulations = 10_000_000
time_steps = 1
spot, strike, rate, vol, T = 100.0, 100.0, 0.05, 0.90, 1.0

# Initialize C++ pricer and Geometric Brownian Motion model
pricer = montecarlo_cpp.MonteCarloPricer(simulations, time_steps, T)
gbm = montecarlo_cpp.GeometricBrownianMotion(rate, vol)

# Execute simulation paths and extract terminal underlying prices (ST)
raw_paths = pricer.generate_paths(spot, gbm)
paths = np.array(raw_paths).reshape(simulations, time_steps + 1)
ST = paths[:, -1]

discount = np.exp(-rate * T)

# Option payoffs and premium estimation
call_payoffs = np.maximum(ST - strike, 0.0) * discount
premium = np.mean(call_payoffs)

# PnL profiles
pnl_short = premium - call_payoffs
pnl_long = call_payoffs - premium


losses_short = -pnl_short
var_95 = np.percentile(losses_short, 95)
var_99 = np.percentile(losses_short, 99)
es_95 = np.mean(losses_short[losses_short >= var_95])


pnl_var_95, pnl_var_99, pnl_es_95 = -var_95, -var_99, -es_95

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

# Short call P & L distribution
fig1, ax1 = plt.subplots(figsize=(5, 3.8), dpi=300)

counts, bins, _ = ax1.hist(pnl_short, bins=120, density=True, alpha=0.6, color='#2b5c8f')
ax1.axvline(premium, color='green', linestyle='--', linewidth=1.2, label=f'Max Profit ({premium:.2f} €)')
ax1.axvline(pnl_var_95, color='#d95f02', linestyle='-', linewidth=1.2, label=f'95% VaR ({pnl_var_95:.2f} €)')

ax1.set_title('Short Call Full P&L Distribution', fontsize=10, fontweight='bold')
ax1.set_xlabel('P&L (€)', fontsize=9)
ax1.set_ylabel('Density', fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('short_call_full.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# VaR and ES

fig2, ax2 = plt.subplots(figsize=(5, 3.8), dpi=300)

# Masking bins to dynamically adapt the vertical axis on the tail
mask_tail = bins[:-1] <= pnl_var_95 + 5
ax2.hist(pnl_short, bins=150, density=True, alpha=0.6, color='#2b5c8f')

# 95% Expected Shortfall region
ax2.axvspan(bins[0], pnl_var_95, alpha=0.25, color='#e7298a', label='95% ES Area')

# Risk thresholds
ax2.axvline(pnl_var_95, color='#d95f02', linestyle='-', linewidth=1.5, label=f'95% VaR ({pnl_var_95:.2f} €)')
ax2.axvline(pnl_var_99, color='#7570b3', linestyle='-', linewidth=1.5, label=f'99% VaR ({pnl_var_99:.2f} €)')
ax2.axvline(pnl_es_95, color='#e7298a', linestyle=':', linewidth=1.8, label=f'95% ES ({pnl_es_95:.2f} €)')


ax2.set_xlim(bins[0], pnl_var_95 + 8)
ax2.set_ylim(0, np.max(counts[mask_tail]) * 1.15)

ax2.set_title('Left-Tail Risk Focus (Losses)', fontsize=10, fontweight='bold')
ax2.set_xlabel('P&L (€)', fontsize=9)
ax2.set_ylabel('Density', fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('short_call_tail.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# Long call figure

fig3, ax3 = plt.subplots(figsize=(5, 3.8), dpi=300)

ax3.hist(pnl_long, bins=120, density=True, alpha=0.6, color='#1b7837')
ax3.axvline(-premium, color='red', linestyle='--', linewidth=1.2, label=f'Max Loss (-{premium:.2f} €)')

ax3.set_title('Long Call Full P&L Distribution', fontsize=10, fontweight='bold')
ax3.set_xlabel('P&L (€)', fontsize=9)
ax3.set_ylabel('Density', fontsize=9)
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('long_call_full.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
