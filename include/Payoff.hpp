#ifndef PAYOFF_HPP
#define PAYOFF_HPP

#include <algorithm> // Necessario per usare std::max

// Interfaccia astratta per il Payoff
class Payoff {
public:
    virtual double operator()(double spot) const = 0;
    virtual ~Payoff() = default;
};

// Payoff per un'opzione Call: max(S - K, 0)
class CallPayoff : public Payoff {
private:
    double strike;
public:
    CallPayoff(double K) : strike(K) {}
    double operator()(double spot) const override {
        return std::max(spot - strike, 0.0);
    }
};

// Payoff per un'opzione Put: max(K - S, 0)
class PutPayoff : public Payoff {
private:
    double strike;
public:
    PutPayoff(double K) : strike(K) {}
    double operator()(double spot) const override {
        return std::max(strike - spot, 0.0);
    }
};

class AsianCallPayoff : public Payoff {
private:
    double strike;
public:
    AsianCallPayoff(double K) : strike(K) {}
    
    // Calcola il payoff basandosi sulla media passata come argomento
    double operator()(double average_price) const override {
        return std::max(average_price - strike, 0.0);
    }
};

#endif // PAYOFF_HPP

