# Screening
There are four criteria for outlier detection

(1) DC: Direction changes (slope & curvature)
(2) BC: Big changes (slope & curvature)
(3) BC_slope: Big changes (only slope)
(4) TA: Taylor approximation

- DC is always on, because it only relies on differences of the numerator, i.e. the denominator will not affect the results.

- BC heavily relies on the denominator, especially for the second deriative. Since we only use backward differences (not central differences) for the derivatives, their accuracy is not very high. Even with np.gradient there is still a high dependancy from the denominator, which will then falsely predict an outlier, just because two points are very close to each other.

- BC_slope: Therefore, to counter false positives that are caused by the second derivative, we also track the big changes only of the slope.

- TA: Is disregarded, as it has too many false positives.


# Fine Tuning (check influence of BC_curvature)
(1) Detect all outliers DC + BC
(2) Note that BC >= BC_slope
(3) Check difference |'DC+BC' - 'DC+BC_slope'|
(4) Filter out 0-difference (they are irrelevant)
(5) Plot the systems with diff = [1, 2, 3, ...]
(6) Differences are only caused by BC_curvature
(7) Decide where to switch from BC to BC_slope
(8) Manually update 'Outlier' accordingly

In summary: Use DC + BC as default and afterwards check their differences and decide whether to switch from BC to BC_slope if BC produced false positives.

# Close gap
