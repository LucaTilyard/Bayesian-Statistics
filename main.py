import matplotlib.pyplot as plt
import numpy as np

# Settings
RANDOM_Y_ERROR = True

# Create random linear data
m = 5
c = 3
y_error = 2.5

xs = np.linspace(0, 10, 10)
ys = m * xs + c


if RANDOM_Y_ERROR:
    y_errors = []
    for i in range(len(xs)):
        y_errors.append(abs(np.random.normal(y_error, 1)))
else:
    y_errors = y_error*np.ones(len(xs))

# Apply random Gaussian noise to data points to simulate measurement errors
ys_noisy =[]
for i in range(len(xs)):
    ys_noisy.append(ys[i] + np.random.normal(0, y_error))

def chi_squared(parameters, y_errors):
    """
    Calculate the chi-squared statistic for a linear model y = mx + c.

    Parameters
    ----------
    parameters : np.array of shape (2,)
        Model parameters where parameters[0] is the gradient m
        and parameters[1] is the intercept c.

    Returns
    -------
    chi2 : float
        The chi-squared value of the model fit to the data.
    """
    m = parameters[0]
    c = parameters[1]
    chi2 = 0
    for i in range(len(xs)):
        y_model = m * xs[i] + c
        chi2 += ((ys_noisy[i] - y_model) ** 2) / (y_errors[i] ** 2)
    return chi2

def MCMC(depth, initial_parameters, y_errors):
    """
    Run a Markov Chain Monte Carlo search for the best-fit linear model parameters.

    Parameters
    ----------
    depth : int
        Number of MCMC iterations to perform.

    initial_parameters : array_like of shape (2,)
        Starting point for the chain where initial_parameters[0] is the
        gradient m and initial_parameters[1] is the intercept c.

    Returns
    -------
     chain : np.ndarray of shape (depth + 1, 2)
        Full parameter chain where each row is [m, c] at that iteration.
        chain[-1] is the final accepted state.
    """
    currant_depth = 0
    chain = np.empty((depth + 1, 2), dtype=float)
    chain[0] = np.array(initial_parameters, dtype=float)

    while currant_depth < depth:
        print("MCMC depth: ", currant_depth)
        currant_depth += 1

        # Generate proposal position using multivariate normal distribution
        proposed_parameters = np.random.multivariate_normal(chain[currant_depth-1], np.eye(2))

        # Calculate and compare chi^2 for proposed parameters
        if chi_squared(proposed_parameters, y_errors) < chi_squared(chain[currant_depth-1],y_errors):
            chain[currant_depth] = proposed_parameters
        else:
            chain[currant_depth] = chain[currant_depth-1]
    return chain

# Run chain to fit parameters to data
chain = MCMC(10000, [30, 30], y_errors)

calculated_m = chain[-1][0]
calculated_c = chain[-1][1]

y_line_estimated = calculated_m * xs + calculated_c

# Plot fitted line, true line, and noisy points (+ error bars) on same graph
plt.figure()
plt.errorbar(xs, ys_noisy, fmt="x", yerr=y_errors, capsize=5, linestyle="none", label="Data")
plt.plot(xs, y_line_estimated, "r-", label=f"Fit: y = {calculated_m:.3f}x + {calculated_c:.3f}")
plt.plot(xs, ys, "b--", label=f"True: y = {m:.3f}x + {c:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Create trance plots
time_steps = np.arange(len(chain))
xs_chain = chain[:, 0]
ys_chain = chain[:, 1]

plt.figure()
plt.plot(time_steps, xs_chain, label="X trace plot")
plt.show()

plt.figure()
plt.plot(time_steps, ys_chain, label="Y trace plot")
plt.show()


# TODO:
# Fix Y_errors generalisation, currently distribution is not totally normal due to abs()
# Create animation of parameters evolving in parameter space
# Fix used of global xs and ys in chi_squared function
# Investigate dip in trace of xs before it returns to the correct value



