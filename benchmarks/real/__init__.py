"""Benchmarks against real measured spectra with independently known truth.

The synthetic suite in :mod:`benchmarks` controls every parameter, which makes it
reproducible but unable to expose behaviour that depends on properties of real
measurements -- spectral resolution above all.  This package complements it with
recorded vibration data whose peak frequencies follow from machine geometry rather
than from anything the detector or the test author chose.

Data is fetched on demand and cached, never bundled: see :mod:`benchmarks.real.cwru`.
"""
