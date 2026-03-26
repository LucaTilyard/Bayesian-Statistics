import matplotlib.pyplot as plt
import numpy as np

# settings
RANDOM_Y_ERROR = False

# create random linear data
m = 1
c = 2
y_error = 0.5

xs = np.linspace(0, 10, 10)
ys = m * xs + c


if RANDOM_Y_ERROR:
    y_errors = []
    for i in range(len(xs)):
        y_errors.append(abs(np.random.normal(y_error, 1)))
else:
    y_errors = y_error*np.ones(len(xs))

# apply random Gaussian noise to data points to simulate measurement errors
ys_noisy =[]

for i in range(len(xs)):
    ys_noisy.append(ys[i] + np.random.normal(0, y_error/2))

def chi_squared(parameters, y_errors):
    """
    Calculate the chi-squared statistic for a linear model y = mx + c.

    Parameters
    -------
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
    -------
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
    accepted_count = 0
    step_scale_factor = 0.005

    chain = np.empty((depth + 1, 2), dtype=float)
    chain[0] = np.array(initial_parameters, dtype=float)

    while currant_depth < depth:
        print("MCMC depth: ", currant_depth)
        currant_depth += 1

        # Generate proposal position using multivariate normal distribution
        proposed_parameters = np.random.multivariate_normal(chain[currant_depth-1], step_scale_factor * np.eye(2))

        # Calculate and compare chi^2 for proposed parameters
        transition_probability = min(1, np.exp(-0.5 * (chi_squared(proposed_parameters, y_errors) - chi_squared(chain[currant_depth-1], y_errors))))
        if np.random.rand() < transition_probability:
            chain[currant_depth] = proposed_parameters
            accepted_count += 1
        else:
            chain[currant_depth] = chain[currant_depth-1]

    acceptance_rate = accepted_count / depth
    print(f"Final Acceptance Rate: {acceptance_rate:.2%}")
    return chain

# run chain to fit parameters to data
chain = MCMC(100000, [5, 5], y_errors)

calculated_m = chain[-1][0]
calculated_c = chain[-1][1]

y_line_estimated = calculated_m * xs + calculated_c

# plot fitted line, true line, and noisy points (+ error bars) on same graph
plt.figure()
plt.errorbar(xs, ys_noisy, fmt="x", yerr=y_errors, capsize=5, linestyle="none", label="Data")
plt.plot(xs, y_line_estimated, "r-", label=f"Fit: y = {calculated_m:.3f}x + {calculated_c:.3f}")
plt.plot(xs, ys, "b--", label=f"True: y = {m:.3f}x + {c:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# create trance plots
time_steps = np.arange(len(chain))
xs_chain = chain[:, 0]
ys_chain = chain[:, 1]

plt.figure()
plt.plot(time_steps, xs_chain, label="X trace plot")
plt.show()

plt.figure()
plt.plot(time_steps, ys_chain, label="Y trace plot")
plt.show()

print("c", calculated_c, "m", calculated_m)

burn_in = int(0.2 * len(chain))
valid_chain = chain[burn_in:]

calculated_m = np.mean(valid_chain[:, 0])
calculated_c = np.mean(valid_chain[:, 1])
error_m = np.std(valid_chain[:, 0])
error_c = np.std(valid_chain[:, 1])
print(f"m = {calculated_m:.3f} +/- {error_m:.3f}")
print(f"c = {calculated_c:.3f} +/- {error_c:.3f}")

# plot chain probibilit distributions
plt.figure()
plt.hist(valid_chain[:, 0], bins=30, density=True, alpha=0.7, label="m distribution")
plt.axvline(m, color="r", linestyle="--", label=f"True m = {m}")
plt.xlabel("m")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()  

plt.figure()
plt.hist(valid_chain[:, 1], bins=30, density=True, alpha=0.7, label="c distribution")
plt.axvline(c, color="r", linestyle="--", label=f"True c = {c}")
plt.xlabel("c")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()  

# create a heatmap of parameter space 
m_values = np.linspace(calculated_m - 3*error_m, calculated_m + 3*error_m, 100)
c_values = np.linspace(calculated_c - 3*error_c, calculated_c + 3*error_c, 100)
chi2_values = np.empty((len(m_values), len(c_values)))
for i in range(len(m_values)):
    for j in range(len(c_values)):
        chi2_values[i, j] = chi_squared([m_values[i], c_values[j]], y_errors)
plt.figure()
plt.imshow(chi2_values.T, extent=(m_values[0], m_values[-1], c_values[0], c_values[-1]), origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Chi-squared")
plt.xlabel("m")
plt.ylabel("c")
plt.title("Chi-squared Heatmap")
plt.scatter(calculated_m, calculated_c, color="r", label="Estimated parameters")
plt.scatter(m, c, color="w", label="True parameters")
plt.legend()
plt.show()

# create phase diagram of parameter space
plt.figure()
plt.scatter(chain[:, 0], chain[:, 1], alpha=0.5, s=10, label="MCMC samples")
plt.xlabel("m")
plt.ylabel("c")
plt.title("Parameter Space")
plt.axvline(m, color="r", linestyle="--", label=f"True m = {m}")
plt.axhline(c, color="r", linestyle="--", label=f"True c = {c}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import corner

# valid_chain should be an array of shape (N_steps, 2)
fig = corner.corner(
    valid_chain, 
    labels=["Slope (m)", "Intercept (c)"],
    truths=[1.0, 2.0],
    quantiles=[0.16, 0.5, 0.84], 
    show_titles=True, 
    title_kwargs={"fontsize": 12}
)
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# plot the m trace
axes[0].plot(chain[:, 0], color="k", alpha=0.5)
axes[0].set_ylabel("Slope (m)")
axes[0].axvline(x=burn_in, color="r", linestyle="--", label="End of Burn-in")
axes[0].legend()

# plot the c trace
axes[1].plot(chain[:, 1], color="k", alpha=0.5)
axes[1].set_ylabel("Intercept (c)")
axes[1].set_xlabel("Step Number")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))


plt.errorbar(xs, ys_noisy, yerr=y_errors, fmt=".k", capsize=3, label="Data")


indices = np.random.randint(0, len(valid_chain), size=100)

for idx in indices:
    sample_m = valid_chain[idx, 0]
    sample_c = valid_chain[idx, 1]
    

    plt.plot(xs, sample_m * xs + sample_c, color="red", alpha=0.05)


plt.plot(xs, m * xs + c, "b--", label="True Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Posterior Predictive Check")
plt.legend()
plt.show()

"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# 1. SPEED BOOST: Increase thinning factor to take larger strides
# Now taking every 50th step (200 frames total instead of 500)
thinning_factor = 50
thinned_chain = chain[::thinning_factor]

fig, ax = plt.subplots()

# Set axes limits based on your calculated errors
ax.set_xlim(calculated_m - 4*error_m, calculated_m + 4*error_m)
ax.set_ylim(calculated_c - 4*error_c, calculated_c + 4*error_c)
ax.set_xlabel("m")
ax.set_ylabel("c")
ax.set_title("MCMC Parameter Space Exploration")

# The true parameter red dot has been removed!

# Set up the trail and current point
trail, = ax.plot([], [], "b.-", alpha=0.3, markersize=3, linewidth=0.5, label="MCMC trace") 
current_point, = ax.plot([], [], "ro", markersize=6, label="Current state")

ax.legend()

def update(frame):
    # Plot all points up to the current frame for the faint trail
    trail.set_data(thinned_chain[:frame, 0], thinned_chain[:frame, 1])
    
    # Wrap the single coordinates in lists to satisfy set_data()
    current_point.set_data([thinned_chain[frame, 0]], [thinned_chain[frame, 1]])
    
    return trail, current_point

print("Rendering GIF... this will be much faster now.")

# Create the animation
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=len(thinned_chain), 
    blit=True, 
    repeat=False
)

# 2. SPEED BOOST: Save the animation at 60 FPS instead of 30
ani.save("mcmc_parameter_evolution.mp4", writer="ffmpeg", fps=60)
print("Saved successfully to mcmc_parameter_evolution.mp4!")

plt.show()
""" 
# TODO:
# fix Y_errors generalisation, currently distribution is not totally normal due to abs()
# create animation of parameters evolving in parameter space
# fix used of global xs and ys in chi_squared function
# investigate dip in trace of xs before it returns to the correct value
# missing uncertainties — comapre raio to random number, allows for exploration of paremter space. this allows for uncertainty caculation.
# Calculate and plot prosteriors
# Mention the relation of the parameter 

# Create a heatmap of parameter space 