# Third-party code

## pocketfft (`pocketfft_hdronly.h`)

Header-only C++ FFT library by Martin Reinecke and Peter Bell (Max-Planck-Society), BSD 3-clause licence (the full
notice is at the top of the header), the C++ pocketfft that numpy uses. Upstream: the `cpp` branch of
<https://github.com/mreineck/pocketfft>. The file is unmodified.

- sha256: `4983075ffefbeaf02e97c9aaac36df44fb59b977ac8bf38c9efdedd9ba3e63bd` (3578 lines)
- Vendored in M5 (2026-11-04), downloaded by the project owner; the upstream commit was not recorded.
- Built with `POCKETFFT_NO_MULTITHREADING` (the kernels are single-threaded; parallel rendering uses worker processes)
  and `POCKETFFT_CACHE_SIZE=64` (plans cached per length), see `CMakeLists.txt`.
- Used through `cpp/fft.hpp` only.
