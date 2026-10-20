| Geometry | Target | Design | R² | RMSE | MAE | MAPE % | Median rel. err. % | 95th pct rel. err. % | Max rel. err. % | Within 0.1 % | Within 0.5 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cone | Nu | PASR quad anchor | 0.9999999 ± 0.0000000 | 1.08e-04 ± 5.48e-06 | 6.60e-05 ± 4.04e-06 | 0.015 ± 0.001 | 0.009 ± 0.001 | 0.043 ± 0.002 | 0.687 ± 0.464 | 99.2 ± 0.3 | 99.9 ± 0.1 |
| cone | Sh | PASR quad anchor + log(Nt/Nb) | 0.9999962 ± 0.0000027 | 4.10e-04 ± 1.38e-04 | 1.87e-04 ± 1.50e-05 | 0.012 ± 0.001 | 0.006 ± 0.000 | 0.036 ± 0.004 | 0.514 ± 0.183 | 99.0 ± 0.3 | 99.9 ± 0.0 |
| plate | Nu | PASR quad anchor | 0.9999998 ± 0.0000000 | 8.98e-05 ± 1.29e-05 | 5.03e-05 ± 3.29e-06 | 0.015 ± 0.001 | 0.009 ± 0.000 | 0.043 ± 0.002 | 0.598 ± 0.307 | 99.2 ± 0.3 | 99.9 ± 0.1 |
| plate | Sh | PASR quad anchor + log(Nt/Nb) | 0.9999972 ± 0.0000022 | 3.26e-04 ± 1.25e-04 | 1.28e-04 ± 1.92e-05 | 0.008 ± 0.001 | 0.004 ± 0.000 | 0.026 ± 0.003 | 0.373 ± 0.117 | 99.5 ± 0.2 | 100.0 ± 0.0 |
| wedge | Nu | PASR quad anchor | 0.9999998 ± 0.0000001 | 9.25e-05 ± 1.58e-05 | 5.03e-05 ± 3.36e-06 | 0.015 ± 0.001 | 0.009 ± 0.000 | 0.043 ± 0.001 | 0.573 ± 0.276 | 99.1 ± 0.3 | 99.9 ± 0.0 |
| wedge | Sh | PASR quad anchor + log(Nt/Nb) | 0.9999971 ± 0.0000022 | 3.36e-04 ± 1.17e-04 | 1.30e-04 ± 1.77e-05 | 0.008 ± 0.001 | 0.004 ± 0.000 | 0.025 ± 0.003 | 0.367 ± 0.107 | 99.4 ± 0.2 | 100.0 ± 0.0 |

Test-set accuracy of the final surrogates, mean ± sd over five random 70/10/20 splits (split seeds 5–9, 1,100 test rows each), each model a 3-network ensemble. RMSE and MAE are absolute (dimensionless Nu, Sh). No design or setting was chosen using these test sets.
