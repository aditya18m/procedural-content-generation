# Procedural Terrain Generator

## Features

- Interactive terrain generation with customizable water, sand, greenery, and ice percentage
- 2D biome map visualization
- 3D terrain visualization
- Erosion simulation to add more natural-looking terrain
- Comprehensive unit tests to ensure the correctness of the terrain generation process

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- SciPy

You can install the required dependencies using pip:

```
pip install numpy matplotlib scipy
```

### Running the Terrain Generator

To run the interactive terrain generator, execute the following command in your terminal:

```
python pcg.py
```

This will open an interactive interface where you can enter the desired percentages for water, sand, greenery, and ice within the land. The program will then generate a 2D biome map and a 3D terrain visualization.

### Running the Tests

To run the unit tests for the terrain generation functions, use the following command:

```
python -m unittest pcgTests.py
```

This will execute the test suite defined in the `pcgTests.py` file and display the results.

## Project Structure

- `pcg.py`: The main script that contains the terrain generation functions and the interactive interface.
- `pcgTests.py`: The test suite for the terrain generation functions.
