
#ifndef PROCESS_HPP
#define PROCESS_HPP
// Abstract Interface
class StochasticProcess {
    public:
    virtual double evolve(double current_spot, double dt, double random_gaussian) const = 0;
    virtual double get_drift() const = 0;
    virtual double get_volatility() const = 0;
    virtual ~StochasticProcess() = default;
};
// Derived class for Geometric Brownian Motion (GBM)
class GeometricBrownianMotion : public StochasticProcess {
    private:
        double drift;
        double volatility;
public:
    GeometricBrownianMotion(double r, double vol);

    double evolve(double current_spot, double dt, double random_gaussian) const override;
    [[nodiscard]] double get_drift() const override;
    [[nodiscard]] double get_volatility() const override;
};
#endif

