# SpeedBench

[![CI](https://github.com/kapildhinwa2010-dev/SpeedBench/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SpeedBench/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/SpeedBench/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SpeedBench/actions/workflows/codeql.yml)

Wall-clock timing of compression operations (SVD, quantize, Fourier, hybrid) by matrix size.

## What it does

Times four compression operations on random square matrices of several
sizes using time.perf_counter with best-of-N repeats, and returns one
row per (size, method) with the measured seconds.

    from speedbench import speed_table

    for row in speed_table(sizes=(64, 128, 256), repeats=3):
        print(row["size"], row["method"], row["seconds"])

## API

- time_operation(func, repeats) - best-of-N wall-clock seconds
- speed_table(sizes, repeats, seed) - one row per size and method

## Honest results and limitations

- Timings are wall-clock on the machine running the benchmark and vary
  with BLAS backend, CPU load and numpy build. Nothing here is a
  portable speedup claim; compare only rows within one run.
- Best-of-N damps scheduler noise but biases optimistic; medians would be
  more conservative.
- Matrices are random: timings reflect shapes, not structure.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
