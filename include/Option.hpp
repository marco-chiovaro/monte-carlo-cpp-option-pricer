#ifndef OPTION_HPP
#define OPTION_HPP
	
#include "Payoff.hpp"
#include <vector>
	
// Abstract Base Interface for all Options
class Option {
	protected:
		const Payoff& payoff; // Dependency Injection: Option holds a reference to a Payoff 
		
	public:
		// Constructor taking a reference to the Payoff
		Option(const Payoff& p) : payoff(p) {}
	
		virtual double get_payoff(const std::vector<double>& path) const = 0;
		
		virtual ~Option() = default;
};

	
class EuropeanOption : public Option {
	public:
		EuropeanOption(const Payoff& p) : Option(p) {}
		
		double get_payoff(const std::vector<double>& path) const override {
			return payoff(path.back()); 
		}
};
	
	
class AsianOption : public Option {
	public:
		AsianOption(const Payoff& p) : Option(p) {}
		
		double get_payoff(const std::vector<double>& path) const override {
			double sum = 0.0;
			for (double price : path) {
				sum += price;
			}
			double average = sum / path.size();
			return payoff(average);
		}
};
	
#endif 	
