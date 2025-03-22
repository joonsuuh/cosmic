# COSMIC OpenGL Simulated Metric In CUDA (COSMIC)

Raytracing a blackhole image (or video if real-time is possible on gpu)

Features:

- [x] explicit RK45 ODE Solver (Dormand-Prince adaptive step size)
- [x] Kerr Metric (rotating black hole) using Boyer-Lindquist coordinates
- [x] 3 + 1 Formulism Hamiltonian to decompose spacetime into equations of motion for the photon
- [x] Doppler boosting & gravitational redshift
- [x] OpenMP parallelization on CPU
- [ ] OpenGL rendering
- [ ] CUDA accelerated computations
- [ ] Metal acceleration for macOS?

TODO:

- [ ] CUDA parallelization on GPU
- [ ] clean up cpu implementation (run opengl on mac/windows/linux)
- [ ] cuda implementation
- [ ] write-up or video???
