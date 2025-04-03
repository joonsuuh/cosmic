#ifndef COSMIC_DORMAND_PRINCE_CUDA_CUH
#define COSMIC_DORMAND_PRINCE_CUDA_CUH

#include <cuda_runtime.h>
#include <cmath>

// CUDA-optimized version of the Dormand-Prince integrator
class CudaDormandPrinceRK45 {
  public:
  __device__ CudaDormandPrinceRK45(int num_equations, float tolerance_abs, float tolerance_rel,
    float hmin, float hmax, float initial_step,
    int max_iterations) 
    : N{num_equations}
    , atol{tolerance_abs}
    , rtol{tolerance_rel}
    , hmin{hmin}
    , hmax{hmax}
    , initial_step{initial_step}
    , max_iterations{max_iterations}
    {
      // Use shared or register memory in CUDA kernel context
      // This constructor should be called in device code
    }
    
    // Custom inline functions for performance
    __device__ inline float max_val(float a, float b) {
      return a > b ? a : b;
    }
    
    __device__ inline float clamp_val(float value, float min_val, float max_val) {
      return value < min_val ? min_val : (value > max_val ? max_val : value);
    }
    
    __device__ inline float min_val(float a, float b) {
      return a < b ? a : b;
    }
    
    // CUDA optimized integrator for device execution
    template<typename DerivativeFunctor, typename HitFunctor, typename BoundaryFunctor>
    __device__ bool integrate(DerivativeFunctor& derivFunctor, 
      HitFunctor& hitFunctor,
      BoundaryFunctor& boundaryFunctor,
      float* y,
      float* k1, float* k2, float* k3, float* k4, 
      float* k5, float* k6, float* k7,
      float* y_err, float* y_next) {
        float h = initial_step;
        float err = 1.0f;
        float err_prev = 1.0f;
        
        // First same as last (FSAL) so k1 <- k7 
        derivFunctor(y, k1);
        
        // Limit maximum iterations to prevent GPU kernel hangs
        int iter = 0;
        
        while (!boundaryFunctor(y) && iter++ < max_iterations) {
          step(derivFunctor, h, y, k1, k2, k3, k4, k5, k6, k7, y_err, y_next);
          err = max_val(error(y_next, y_err), hmin);
          
          if (err < 1.0f) {
            // Use loop for small arrays (faster than memcpy for few elements)
            for (int i = 0; i < N; ++i) {
              y[i] = y_next[i];
            }
            
            // swap k1 and k7 pointers
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
          if (err_prev < 1.0f) {
            // PI controller step size adjustment for good step size
            h = h * S * powf(err, -alpha) * powf(err_prev, beta);
          } else {
            // Reject step and reduce step size
            h = min_val(h * S / powf(err, 0.2f), h);
          }
          h = clamp_val(h, hmin, hmax);
        }
        
        return false; // Return false if no hit detected or max iterations reached
      }
      
      // CUDA optimized step function
      template <typename DerivativeFunction>
      __device__ void step(DerivativeFunction& derivFunc, float h, const float* y,
        float* k1, float* k2, float* k3, float* k4, 
        float* k5, float* k6, float* k7,
        float* y_err, float* y_next) {  
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
            
            __device__ float error(const float* y_next, const float* y_err) {
              float err_sum = 0.0f;
              for (int i = 0; i < N; i++) {
                float e = y_err[i] / (atol + max_val(fabsf(y_next[i]), fabsf(y_next[i])) * rtol);
                err_sum += e * e;
              }
              return sqrtf(err_sum / N);
            }
            
            private:
            // initial stuff
            const int N;
            const float atol, rtol;
            
            // clamp values
            const float hmin;
            const float hmax;
            const float initial_step;
            const int max_iterations;
            
            // Adaptive step size constants
            static constexpr float S = 0.9f;  // Safety factor
            static constexpr float alpha = 0.7f/5.0f;  // PI controller parameters
            static constexpr float beta = 0.4f/5.0f;
            
            // Butcher tableau constants
            static constexpr float c2 = 1.0f / 5.0f;
            static constexpr float c3 = 3.0f / 10.0f;
            static constexpr float c4 = 4.0f / 5.0f;
            static constexpr float c5 = 8.0f / 9.0f;
            
            static constexpr float a21 = 1.0f / 5.0f;
            static constexpr float a31 = 3.0f / 40.0f;
            static constexpr float a32 = 9.0f / 40.0f;
            static constexpr float a41 = 44.0f / 45.0f;
            static constexpr float a42 = -56.0f / 15.0f;
            static constexpr float a43 = 32.0f / 9.0f;
            static constexpr float a51 = 19372.0f / 6561.0f;
            static constexpr float a52 = -25360.0f / 2187.0f;
            static constexpr float a53 = 64448.0f / 6561.0f;
            static constexpr float a54 = -212.0f / 729.0f;
            static constexpr float a61 = 9017.0f / 3168.0f;
            static constexpr float a62 = -355.0f / 33.0f;
            static constexpr float a63 = 46732.0f / 5247.0f;
            static constexpr float a64 = 49.0f / 176.0f;
            static constexpr float a65 = -5103.0f / 18656.0f;
            
            static constexpr float a71 = 35.0f / 384.0f;
            static constexpr float a72 = 0.0f;
            static constexpr float a73 = 500.0f / 1113.0f;
            static constexpr float a74 = 125.0f / 192.0f;
            static constexpr float a75 = -2187.0f / 6784.0f;
            static constexpr float a76 = 11.0f / 84.0f;
            
            static constexpr float e1 = 71.0f / 57600.0f;
            static constexpr float e2 = 0.0f;
            static constexpr float e3 = -71.0f / 16695.0f;
            static constexpr float e4 = 71.0f / 1920.0f;
            static constexpr float e5 = -17253.0f / 339200.0f;
            static constexpr float e6 = 22.0f / 525.0f;
            static constexpr float e7 = -1.0f / 40.0f;
          };
          
          #endif // COSMIC_DORMAND_PRINCE_CUDA_CUH
          