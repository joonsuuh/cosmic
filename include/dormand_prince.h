#ifndef COSMIC_DORMAND_PRINCE_H
#define COSMIC_DORMAND_PRINCE_H

#include <cmath>

class DormandPrinceRK45 {
public:
    DormandPrinceRK45(int num_equations, float tolerance_abs, float tolerance_rel,
                      float hmin, float hmax, float initial_step, int max_iter)
      : N(num_equations)
      , atol(tolerance_abs)
      , rtol(tolerance_rel)
      , hmin(hmin)
      , hmax(hmax)
      , h(initial_step)
      , max_iter(max_iter)
      // Raw pointer allocation
      , k1 {new float[static_cast<size_t>(N)]{}}
      , k2 {new float[static_cast<size_t>(N)]{}}
        , k3 {new float[static_cast<size_t>(N)]{}}
        , k4 {new float[static_cast<size_t>(N)]{}}
        , k5 {new float[static_cast<size_t>(N)]{}}
        , k6 {new float[static_cast<size_t>(N)]{}}
        , k7 {new float[static_cast<size_t>(N)]{}}
        , y_err {new float[static_cast<size_t>(N)]{}}
        , y_next {new float[static_cast<size_t>(N)]{}}

    {      
    }

    // Destructor to clean up allocated memory
    ~DormandPrinceRK45() {
      delete[] k1;
      delete[] k2;
      delete[] k3;
      delete[] k4;
      delete[] k5;
      delete[] k6;
      delete[] k7;
      delete[] y_err;
      delete[] y_next;
    }

    DormandPrinceRK45(const DormandPrinceRK45&) = delete;
    DormandPrinceRK45& operator=(const DormandPrinceRK45&) = delete;
    DormandPrinceRK45(DormandPrinceRK45&&) = delete;
    DormandPrinceRK45& operator=(DormandPrinceRK45&&) = delete;

    // Custom inline functions for performance
    inline float max_val(float a, float b) {
        return a > b ? a : b;
    }
    
    inline float clamp_val(float value, float min_val, float max_val) {
        return value < min_val ? min_val : (value > max_val ? max_val : value);
    }

    inline float min_val(float a, float b) {
        return a < b ? a : b;
    }

    // integrates ODEs and checks for hit on the disk
    // returns if hit detected
    template<typename DerivativeFunctor, typename HitFunctor, typename BoundaryFunctor>
    bool integrate(DerivativeFunctor& derivFunctor, 
                  HitFunctor& hitFunctor,
                  BoundaryFunctor& boundaryFunctor,
                  float* y) {
        h = 0.1;
        float err = 1.0f;
        float err_prev = 1.0f;
        // bool hit_detected = false; // Local variable instead of class member

        // First same as last (FSAL) so k1 <- k7 
        derivFunctor(y, k1);
        
	int iterations {0};

        while (!boundaryFunctor(y)) {
	    iterations++;
            step(derivFunctor, h, y);
            err = max_val(error(), hmin);
            
            if (err < 1.0) {
                for (int i = 0; i < N; ++i) {
                    y[i] = y_next[i];
                }
                
                // raw pointer swap for k1 <- k7
                float* temp = k1;
                k1 = k7;
                k7 = temp;

                // for PI_controller
                err_prev = err;
                
                // Check if we hit the disk after this step
                if (hitFunctor(y)) {
                    return true; // Return immediately if hit detected
                }
            }
            if (err_prev < 1.0) {
                // PI controller step size adjustment for good step size
                h = h * S * std::powf(err, -alpha) * std::powf(err_prev, beta);
            } else {
                // Reject step and reduce step size
                h = min_val(h * S / std::powf(err, 0.2), h);
            }
            h = clamp_val(h, hmin, hmax);
        }
	// std::cout << "Iterations: " << iterations << std::endl;
        return false; // Return false if no hit detected
    }

    template <typename DerivativeFunction>
    void step(DerivativeFunction& derivFunc, float h, const float* y) {  // Removed const from h
        // step 2 (step 1 is already done outside)
        for(int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * a21 * k1[i];
        }
        derivFunc(y_next, k2);
        
        // step 3
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        }
        derivFunc(y_next, k3);
        
        // step 4
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        derivFunc(y_next, k4);

        // step 5
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        derivFunc(y_next, k5);

        // step 6
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        derivFunc(y_next, k6);

        // step 7 finally calculate y_next
        for(int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + 
                                   a75 * k5[i] + a76 * k6[i]);
        }
        // Then use y_next to calculate k7
        derivFunc(y_next, k7);

        // find the error
        for(int i = 0; i < N; i++) {
            y_err[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + 
                            e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        }
    }

    float error() {
        float err_sum = 0.0;
        for (int i = 0; i < N; i++) {
            float e = y_err[i] / (atol + max_val(std::fabs(y_next[i]), std::fabs(y_next[i])) * rtol);
            err_sum += e * e;
        }
        return std::sqrtf(err_sum / N);
    }

private:
    // initial stuff
    const int N;
    const float atol, rtol;

    // dynamic array allocation 
    float* k1{}; // same as k1{nullptr} for safety
    float* k2{};
    float* k3{};
    float* k4{};
    float* k5{};
    float* k6{};
    float* k7{};
    float* y_err{};
    float* y_next{};

    // clamp values
    const float hmin{ 1.0e-8f};
    const float hmax{ 0.1f};
    float h{ 0.1f}; // initial step size
    const int max_iter{ 10000}; // max iterations
    // PI controller parameters
    float err{ 1.0f}; // error
    float err_prev{ 1.0f}; // previous error
    
    // Adaptive step size constants
    static constexpr float S = 0.9f;  // Safety factor
    static constexpr float alpha = 0.7/5.0;  // PI controller parameters
    static constexpr float beta = 0.4/5.0;

    // Butcher tableau constants - made static to avoid recalculation
    static constexpr float c2 = 1.0 / 5.0;
    static constexpr float c3 = 3.0 / 10.0;
    static constexpr float c4 = 4.0 / 5.0;
    static constexpr float c5 = 8.0 / 9.0;

    static constexpr float a21 = 1.0 / 5.0;
    static constexpr float a31 = 3.0 / 40.0;
    static constexpr float a32 = 9.0 / 40.0;
    static constexpr float a41 = 44.0 / 45.0;
    static constexpr float a42 = -56.0 / 15.0;
    static constexpr float a43 = 32.0 / 9.0;
    static constexpr float a51 = 19372.0 / 6561.0;
    static constexpr float a52 = -25360.0 / 2187.0;
    static constexpr float a53 = 64448.0 / 6561.0;
    static constexpr float a54 = -212.0 / 729.0;
    static constexpr float a61 = 9017.0 / 3168.0;
    static constexpr float a62 = -355.0 / 33.0;
    static constexpr float a63 = 46732.0 / 5247.0;
    static constexpr float a64 = 49.0 / 176.0;
    static constexpr float a65 = -5103.0 / 18656.0;

    static constexpr float a71 = 35.0 / 384.0;
    static constexpr float a72 = 0.0;
    static constexpr float a73 = 500.0 / 1113.0;
    static constexpr float a74 = 125.0 / 192.0;
    static constexpr float a75 = -2187.0 / 6784.0;
    static constexpr float a76 = 11.0 / 84.0;

    static constexpr float e1 = 71.0 / 57600.0;
    static constexpr float e2 = 0.0;
    static constexpr float e3 = -71.0 / 16695.0;
    static constexpr float e4 = 71.0 / 1920.0;
    static constexpr float e5 = -17253.0 / 339200.0;
    static constexpr float e6 = 22.0 / 525.0;
    static constexpr float e7 = -1.0 / 40.0;
};

#endif // COSMIC_DORMAND_PRINCE_H
