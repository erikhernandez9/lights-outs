import numpy as np

def generate_coefficients_matrix(n):
    """Generate the coefficients matrix for the Lights Out system on an n x n board."""
    size = n * n
    coefficients = np.zeros((size, size), dtype=int)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            coefficients[idx, idx] = 1  # Same light
            if i > 0: coefficients[idx, idx - n] = 1  # Up
            if i < n - 1: coefficients[idx, idx + n] = 1  # Down
            if j > 0: coefficients[idx, idx - 1] = 1  # Left
            if j < n - 1: coefficients[idx, idx + 1] = 1  # Right

    return coefficients

def lights_out_solution_gaussian(board):
    n = board.shape[0]
    size = n * n

    # Generate the coefficients matrix and the results vector (initial state)
    coefficients = generate_coefficients_matrix(n)
    results = board.flatten()

    # Gaussian elimination in modulo 2
    for i in range(size):
        if coefficients[i, i] == 0:
            for j in range(i + 1, size):
                if coefficients[j, i] == 1:
                    coefficients[[i, j]] = coefficients[[j, i]]
                    results[i], results[j] = results[j], results[i]
                    break
        if coefficients[i, i] == 1:
            for j in range(i + 1, size):
                if coefficients[j, i] == 1:
                    coefficients[j] = (coefficients[j] + coefficients[i]) % 2
                    results[j] = (results[j] + results[i]) % 2

    # Back substitution to obtain the solution
    solution = np.zeros(size, dtype=int)
    for i in range(size - 1, -1, -1):
        solution[i] = results[i]
        for j in range(i + 1, size):
            solution[i] = (solution[i] + coefficients[i, j] * solution[j]) % 2

    return solution.reshape((n, n))

def apply_moves(initial_matrix, solution):
    """
    Apply the moves from the solution matrix to the initial matrix.
    Each time a light is pressed, it changes the state of that light and its neighbors (up, down, left, right).
    """
    n = len(initial_matrix)
    resulting_matrix = np.copy(initial_matrix)
    
    # Iterate over the solution matrix to apply the moves
    for i in range(n):
        for j in range(n):
            if solution[i][j] == 1:  # If the solution indicates to press at this position
                # Change the state of the current light
                resulting_matrix[i][j] = 1 - resulting_matrix[i][j]
                # Change the state of the neighbors
                if i > 0:  # Up
                    resulting_matrix[i - 1][j] = 1 - resulting_matrix[i - 1][j]
                if i < n - 1:  # Down
                    resulting_matrix[i + 1][j] = 1 - resulting_matrix[i + 1][j]
                if j > 0:  # Left
                    resulting_matrix[i][j - 1] = 1 - resulting_matrix[i][j - 1]
                if j < n - 1:  # Right
                    resulting_matrix[i][j + 1] = 1 - resulting_matrix[i][j + 1]

    return resulting_matrix

def verify_solution(initial_matrix, solution):
    """
    Verify if the solution is correct for the Lights Out game.
    """
    # Apply the moves from the solution to the initial matrix
    resulting_matrix = apply_moves(initial_matrix, solution)

    # Check if the resulting matrix has all lights off (0s)
    return np.all(resulting_matrix == 0)

# Example usage with an initial 10x10 board
initial_board = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
]
)

solution = lights_out_solution_gaussian(initial_board)
print("Initial board:")
print(initial_board)
print("Proposed solution (press vector):")
print(solution)
if verify_solution(initial_board, solution):
    print("The solution is correct.")
else:
    print("The solution is not correct.")
