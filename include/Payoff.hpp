#ifndef PAYOFF_HPP
#define PAYOFF_HPP
	
#include <algorithm> 
	
// Abstract interface for Option Payoff
class Payoff {
	public:
		virtual double operator()(double spot) const = 0;
		virtual ~Payoff() = default;
};
	
// Call option payoff: max(S - K, 0)
class CallPayoff : public Payoff {
	private:
		double strike;
	public:
		CallPayoff(double K) : strike(K) {}
		double operator()(double spot) const override {
			return std::max(spot - strike, 0.0);
		}
};
	
// Put option payoff: max(K - S, 0)
class PutPayoff : public Payoff {
	private:
		double strike;
	public:
		PutPayoff(double K) : strike(K) {}
		double operator()(double spot) const override {
			return std::max(strike - spot, 0.0);
		}
};

#endif


