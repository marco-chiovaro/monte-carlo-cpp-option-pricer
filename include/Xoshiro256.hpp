#ifndef XOSHIRO256STARSTAR_HPP
#define XOSHIRO256STARSTAR_HPP

#include <cstdint>

class Xoshiro256StarStar {
private:
    uint64_t s[4];

    // Funzione helper per la rotazione dei bit (velocissima sulle CPU)
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    // Tipi e metodi richiesti dalla standard library del C++
    using result_type = uint64_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    // Il costruttore accetta un seed a 64-bit.
    // Utilizza internamente l'algoritmo SplitMix64 per inizializzare 
    // l'array di stato in modo sicuro (Xoshiro non tollera stati con tutti zeri).
    explicit Xoshiro256StarStar(uint64_t seed) {
        uint64_t z = seed;
        for (int i = 0; i < 4; i++) {
            z += 0x9e3779b97f4a7c15;
            uint64_t z_val = z;
            z_val = (z_val ^ (z_val >> 30)) * 0xbf58476d1ce4e5b9;
            z_val = (z_val ^ (z_val >> 27)) * 0x94d049bb133111eb;
            s[i] = z_val ^ (z_val >> 31);
        }
    }

    // L'operatore di estrazione del numero casuale (Il cuore del motore)
    inline result_type operator()() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }
};

#endif // XOSHIRO256STARSTAR_HPP