def standard_monte_carlo(function, samples):
    return 1/len(samples) * sum(function(samples))