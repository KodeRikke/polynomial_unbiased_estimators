# Leading-order noise terms

Difference analyzed: naive - unbiased. Positive values imply unbiased is better for that metric.

**IMPORTANT:** This table shows the leading-order Taylor coefficient at **small noise scales** (t → 0, i.e., very high privacy / low ε). At practical DP noise scales (ε ∈ [0.5, 5]), **higher-order terms often dominate** and can flip the sign. For example, Chebyshev T9 has a negative leading t⁴ term, but a large positive t⁶ term that overwhelms it at t=1. Thus, the sign at practical noise scales may differ from this asymptotic prediction.

Laplace expansion variable: t = Delta/epsilon (with Delta fixed to 1 in expansion).
Gaussian expansion variable: t = sigma.

The sign columns show whether the difference (naive - unbiased) is positive (+) or negative (−) at each q value.

| Polynomial | Noise | Metric diff | Leading term | sign@q=0 | sign@q=0.001 | sign@q=0.1 | sign@q=0.2 | sign@q=0.3 | sign@q=0.4 | sign@q=0.5 | sign@q=1 |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n| quadratic | laplace | naive_mse-unbiased_mse | (4) * noise_scale^4 | + | + | + | + | + | + | + | + |
| quadratic | gaussian | naive_mse-unbiased_mse | (1) * noise_scale^4 | + | + | + | + | + | + | + | + |
| cubic | laplace | naive_mse-unbiased_mse | (108*q**2 + 72*q + 28) * noise_scale^4 | + | + | + | + | + | + | + | + |
| cubic | gaussian | naive_mse-unbiased_mse | (27*q**2 + 18*q + 7) * noise_scale^4 | + | + | + | + | + | + | + | + |
| chebyshev_T3 | laplace | naive_mse-unbiased_mse | (1728*q**2 - 288) * noise_scale^4 | - | - | - | - | - | - | + | + |
| chebyshev_T3 | gaussian | naive_mse-unbiased_mse | (432*q**2 - 72) * noise_scale^4 | - | - | - | - | - | - | + | + |
| cubic_coeffs | laplace | naive_mse-unbiased_mse | (1728*q**2 + 864*q - 156) * noise_scale^4 | - | - | - | + | + | + | + | + |
| cubic_coeffs | gaussian | naive_mse-unbiased_mse | (432*q**2 + 216*q - 39) * noise_scale^4 | - | - | - | + | + | + | + | + |
| high_degree | laplace | naive_mse-unbiased_mse | (36900*q**16 + 59040*q**15 + 69888*q**14 + 72408*q**13 + 69088*q**12 + 61968*q**11 + 52668*q**10 + 42416*q**9 + 32076*q**8 + 22176*q**7 + 12936*q**6 + 7056*q**5 + 3528*q**4 + 1568*q**3 + 588*q**2 + 168*q + 28) * noise_scale^4 | + | + | + | + | + | + | + | + |
| high_degree | gaussian | naive_mse-unbiased_mse | (9225*q**16 + 14760*q**15 + 17472*q**14 + 18102*q**13 + 17272*q**12 + 15492*q**11 + 13167*q**10 + 10604*q**9 + 8019*q**8 + 5544*q**7 + 3234*q**6 + 1764*q**5 + 882*q**4 + 392*q**3 + 147*q**2 + 42*q + 7) * noise_scale^4 | + | + | + | + | + | + | + | + |
| high_degree_coeffs | laplace | naive_mse-unbiased_mse | (147600*q**16 - 354240*q**15 + 675552*q**14 - 1050000*q**13 + 1533880*q**12 - 1943808*q**11 + 2301516*q**10 - 2469976*q**9 + 2469636*q**8 - 2250720*q**7 + 1806600*q**6 - 1323024*q**5 + 863112*q**4 - 491104*q**3 + 223308*q**2 - 78792*q + 15340) * noise_scale^4 | + | + | + | + | + | + | + | + |
| high_degree_coeffs | gaussian | naive_mse-unbiased_mse | (36900*q**16 - 88560*q**15 + 168888*q**14 - 262500*q**13 + 383470*q**12 - 485952*q**11 + 575379*q**10 - 617494*q**9 + 617409*q**8 - 562680*q**7 + 451650*q**6 - 330756*q**5 + 215778*q**4 - 122776*q**3 + 55827*q**2 - 19698*q + 3835) * noise_scale^4 | + | + | + | + | + | + | + | + |
| chebyshev_T9 | laplace | naive_mse-unbiased_mse | (1528823808*q**14 - 4087480320*q**12 + 4208246784*q**10 - 2100142080*q**8 + 523874304*q**6 - 60341760*q**4 + 2488320*q**2 - 25920) * noise_scale^4 | - | - | - | + | - | - | + | + |
| chebyshev_T9 | gaussian | naive_mse-unbiased_mse | (382205952*q**14 - 1021870080*q**12 + 1052061696*q**10 - 525035520*q**8 + 130968576*q**6 - 15085440*q**4 + 622080*q**2 - 6480) * noise_scale^4 | - | - | - | + | - | - | + | + |
