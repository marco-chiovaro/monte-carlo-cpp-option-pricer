
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> 

#include "../include/Process.hpp"
#include "../include/Payoff.hpp"
#include "../include/Option.hpp"   
#include "../include/MonteCarlo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(montecarlo_cpp, m) {
	m.doc() = "High-performance C++ Monte Carlo pricing engine (OOP Architecture)";
	
	// Stochastic process bindings
	
	py::class_<StochasticProcess>(m, "StochasticProcess");
	
	py::class_<GeometricBrownianMotion, StochasticProcess>(m, "GeometricBrownianMotion")
	.def(py::init<double, double>(), py::arg("drift"), py::arg("volatility"))
	.def("evolve", &GeometricBrownianMotion::evolve)
	// Getter methods
	.def("get_drift", &GeometricBrownianMotion::get_drift)
	.def("get_volatility", &GeometricBrownianMotion::get_volatility);
	

	// Payoff bindings

	py::class_<Payoff>(m, "Payoff");
	
	py::class_<CallPayoff, Payoff>(m, "CallPayoff")
	.def(py::init<double>(), py::arg("strike"));
	
	py::class_<PutPayoff, Payoff>(m, "PutPayoff")
	.def(py::init<double>(), py::arg("strike"));
	

	// Option bindings
	
	py::class_<Option>(m, "Option");
	
	py::class_<EuropeanOption, Option>(m, "EuropeanOption")
	.def(py::init<const Payoff&>(), py::arg("payoff"), py::keep_alive<1, 2>()); // Expects a payoff as input
	
	py::class_<AsianOption, Option>(m, "AsianOption")
	.def(py::init<const Payoff&>(), py::arg("payoff"), py::keep_alive<1, 2>()); // Expects a payoff as input
	
	
	// Risk data structure bindings
	
	py::class_<RiskMetrics>(m, "RiskMetrics")
	.def_readonly("price", &RiskMetrics::price)
	.def_readonly("delta", &RiskMetrics::delta)
	.def_readonly("gamma", &RiskMetrics::gamma)
	.def_readonly("vega", &RiskMetrics::vega)
	.def_readonly("theta", &RiskMetrics::theta);
	
	
	//  Monte Carlo pricer bindings
	
	py::class_<MonteCarloPricer>(m, "MonteCarloPricer")
		.def(py::init<long, int, double>(), 
			py::arg("sims"), py::arg("steps"), py::arg("maturity"))
	
	// Expose C++ getters as Python properties
	.def_property_readonly("maturity", &MonteCarloPricer::get_maturity)
	.def_property_readonly("simulations", &MonteCarloPricer::get_simulations)
	.def_property_readonly("steps", &MonteCarloPricer::get_steps)
	
	.def("price", &MonteCarloPricer::price, 
		py::arg("initial_spot"), py::arg("discount_rate"), 
		py::arg("process"), py::arg("option"))
	
	.def("price_batch", &MonteCarloPricer::price_batch,
		py::arg("spots"), py::arg("maturities"), py::arg("r"), 
		py::arg("process"), py::arg("option"),
	"Computes a price array processing vectors in a single C++ call")
	
	
	// Optimized lambda for NumPy 
	.def("generate_paths", [](const MonteCarloPricer& self, double initial_spot, const StochasticProcess& process) {
		
		std::vector<double> flat_paths = self.generate_paths(initial_spot, process);
		
		// Use getters instead of direct member access
		ssize_t rows = self.get_simulations(); 
		ssize_t cols = self.get_steps() + 1;
		
		return py::array_t<double>(
		{rows, cols},        
		flat_paths.data()    
		);
	}, py::arg("initial_spot"), py::arg("process"))
	
	.def("calculate_all_risks", &MonteCarloPricer::calculate_all_risks,
	py::arg("initial_spot"), py::arg("r"), 
	py::arg("process"), py::arg("option"),
	"Computes Price and Greeks in a single optimized pass")
	
	.def("calculate_all_risks_batch", [](const MonteCarloPricer& self, 
	py::array_t<double> spots_array, 
	py::array_t<double> times_array, 
	double r, 
	const GeometricBrownianMotion& process, 
	const Option& option) {
		
		// Direct extraction of memory pointers from NumPy buffers
		py::buffer_info spots_info = spots_array.request();
		py::buffer_info times_info = times_array.request();
		
		double* spots_ptr = static_cast<double*>(spots_info.ptr);
		double* times_ptr = static_cast<double*>(times_info.ptr);
		size_t n = spots_info.size;
		
		// Fast conversion to std::vector for OpenMP processing
		std::vector<double> spots_vec(spots_ptr, spots_ptr + n);
		std::vector<double> times_vec(times_ptr, times_ptr + n);
		
		// Invocation of the parallel C++ engine
		RiskSurface surface = self.calculate_all_risks_batch(spots_vec, times_vec, r, process, option);
		
		// Construct output Python dictionary containing NumPy arrays
		py::dict out;
		out["prices"] = py::array_t<double>(n, surface.prices.data());
		out["deltas"] = py::array_t<double>(n, surface.deltas.data());
		out["gammas"] = py::array_t<double>(n, surface.gammas.data());
		out["vegas"]  = py::array_t<double>(n, surface.vegas.data());
		out["thetas"] = py::array_t<double>(n, surface.thetas.data());
		
		return out;
		
	}, 
	py::arg("spots"), py::arg("maturities"), py::arg("r"), 
	py::arg("process"), py::arg("option"),
	"Computes the Risk Surface in parallel, returning a dict of NumPy arrays");
}
	
