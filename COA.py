import numpy as np
import time

# Chimp Optimization Algorithm
def COA(positions, objective_function, lower_bound, upper_bound, num_iterations):
    [num_chimps, num_dimensions] = positions.shape

    # Initialize the best position and the best objective value
    best_position = positions[0]
    best_value = objective_function(best_position)
    Convergence = np.zeros((1, num_iterations))
    ct = time.time()
    # Iterate through each iteration
    for iteration in range(num_iterations):
        for i in range(num_chimps):
            # Evaluate the objective function at the current position
            current_value = objective_function(positions[i])

            # Update the best position and value if needed
            if current_value < best_value:
                best_value = current_value
                best_position = positions[i]

            # Update the position of the current chimp
            r1 = np.random.random(num_dimensions)
            r2 = np.random.random(num_dimensions)
            positions[i] = best_position - r1 * (best_position - positions[i]) + r2 * (best_position - positions[i])

            # Ensure positions are within bounds
            positions[i] = np.clip(positions[i], lower_bound[i], upper_bound[i])

        Convergence[1, iteration] = best_value
    ct = time.time() - ct
    return best_value, Convergence, best_position, ct

