#ifndef OPTION_HPP
#define OPTION_HPP

#include "Payoff.hpp"
#include <vector>

// Interfaccia Base Astratta per tutte le Opzioni
class Option {
protected:
    const Payoff& payoff; // Dependency Injection: L'opzione "possiede" un Payoff (Call o Put)

public:
    // Costruttore che accetta la referenza al Payoff
    Option(const Payoff& p) : payoff(p) {}

    // Metodo polimorfico puro: calcola il payoff dato l'intero percorso del sottostante
    virtual double get_payoff(const std::vector<double>& path) const = 0;
    
    virtual ~Option() = default;
};

// ---------------------------------------------------------
// OPZIONE EUROPEA: controlla il prezzo al termine della scadenza
// ---------------------------------------------------------
class EuropeanOption : public Option {
public:
    EuropeanOption(const Payoff& p) : Option(p) {}

    double get_payoff(const std::vector<double>& path) const override {
        // Viene preso solo l'ultimo valore
        return payoff(path.back()); 
    }
};

// ---------------------------------------------------------
// OPZIONE ASIATICA: Guarda la media di tutti i prezzi
// ---------------------------------------------------------
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

#endif // OPTION_HPP