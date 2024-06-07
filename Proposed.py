import numpy as np
import time

def PROPOSED(positions, objective_function, lower_bound, upper_bound, num_iterations):
    [num_chimps, num_dimensions] = positions.shape

    # Initialize the best position and the best objective value
    best_position = positions[0]
    fit = np.zeros((num_chimps))
    for i in range(num_chimps):
        fit[i] = objective_function(positions[i, :])
    best_value = objective_function(best_position)
    Convergence = np.zeros((num_iterations))
    ct = time.time()
    # Iterate through each iteration
    for iteration in range(num_iterations):
        for i in range(num_chimps):
            # Update using this below loop
            if (iteration % 5 == 0) and (iteration % 7 == 0):  # get position using the POA
                k = np.random.permutation(num_chimps)
                X_FOOD = positions[k, :]
                F_FOOD = fit[k]
                # PHASE 1: Moving towards prey (exploration phase)
                I = np.round(1 + np.random.rand(1, 1))
                if fit[i] > F_FOOD[i]:
                    X_new = positions[i, :] + np.multiply(np.random.rand(1, 1), (X_FOOD - np.multiply(I, positions[i, :])))
                else:
                    X_new = positions[i, :] + np.multiply(np.random.rand(1, 1), (positions[i, :] - 1.0 * X_FOOD))

                # Updating X_i using (5)
                f_new = objective_function(X_new[i])
                if f_new <= fit[i]:
                    positions[i, :] = X_new[i, :]
                    fit[i] = f_new
                # END PHASE 1: Moving towards prey (exploration phase)
                # PHASE 2: Winging on the water surface (exploitation phase)
                positions[i] = positions[i, :] + np.multiply(
                    np.multiply(0.2 * (1 - iteration / num_iterations), (2 * np.random.rand(1, num_dimensions) - 1)), positions[i, :])

            else:  # get position using the COA
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

        Convergence[iteration] = best_value
    ct = time.time() - ct
    return best_value, Convergence, best_position, ct

