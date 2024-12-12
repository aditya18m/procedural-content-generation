import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, to_rgba

#Perlin Noise
def perlin_2d(width, height, scale=50):
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(hash, x, y):
        h = hash & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

    def noise(x, y):
        X, Y = int(x) & 255, int(y) & 255
        x -= int(x)
        y -= int(y)
        u, v = fade(x), fade(y)
        n00 = grad(p[X + p[Y]], x, y)
        n01 = grad(p[X + p[Y + 1]], x, y - 1)
        n10 = grad(p[X + 1 + p[Y]], x - 1, y)
        n11 = grad(p[X + 1 + p[Y + 1]], x - 1, y - 1)
        n0 = lerp(n00, n10, u)
        n1 = lerp(n01, n11, u)
        return lerp(n0, n1, v)

    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    terrain = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            terrain[i][j] = noise(i / scale, j / scale)

    return (terrain - terrain.min()) / (terrain.max() - terrain.min())

#Worley Noise
def worley_2d(width, height, num_points=10):
    points = np.random.rand(num_points, 2) * [width, height]
    noise = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            min_dist = np.min([np.sqrt((x - i)**2 + (y - j)**2) for x, y in points])
            noise[i][j] = min_dist

    return 1 - (noise / np.max(noise))

#Simulate Erosion
def simulate_erosion(terrain, iterations=5, erosion_factor=0.02):
    for _ in range(iterations):
        erosion = np.random.uniform(-erosion_factor, erosion_factor, terrain.shape)
        terrain += erosion
        terrain = np.clip(terrain, 0, 1)
    return gaussian_filter(terrain, sigma=1)

def classify_biomes(terrain, water_level=0.4, sand_width=0.05, green_max=0.7):
    biomes = np.zeros_like(terrain, dtype=float)
    biomes[terrain <= water_level] = terrain[terrain <= water_level]  #Water gradient
    biomes[(terrain > water_level) & (terrain <= water_level + sand_width)] = 1  #Sand
    biomes[(terrain > water_level + sand_width) & (terrain <= green_max - 0.2)] = 2  #Light greenery
    biomes[(terrain > green_max - 0.2) & (terrain <= green_max)] = 2.5  #Standard greenery
    biomes[(terrain > green_max) & (terrain <= green_max + 0.1)] = 3  #Dark greenery
    biomes[terrain > green_max + 0.1] = 4  # Ice
    return biomes

def visualize_terrain(terrain, biomes, water_level, view_3d=True):
    #Define color maps for water and land
    water_cmap = LinearSegmentedColormap.from_list(
        "WaterGradient", ["#08306b", "#2171b5", "#6baed6"], N=256
    )
    land_cmap = LinearSegmentedColormap.from_list(
        "LandGradient", ["#ffdd99", "#b4d17a", "#33a02c", "#165b33", "#ffffff"], N=256
    )

    x, y = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))
    facecolors = np.zeros(biomes.shape + (4,))  #RGBA shape

    #Apply water gradient
    water_indices = biomes <= water_level
    facecolors[water_indices] = water_cmap(biomes[water_indices] / water_level)

    #Apply land gradient
    land_indices = biomes > water_level
    normalized_land = (biomes[land_indices] - water_level) / (4 - water_level)  #Normalize within land range
    facecolors[land_indices] = land_cmap(normalized_land)

    #Plot 3D or 2D terrain
    fig = plt.figure(figsize=(12, 8))
    if view_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, terrain, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_title("3D Terrain Visualization with Gradients")
    else:
        plt.imshow(facecolors, origin='upper')
        plt.title("2D Biome Map")

    plt.show()


def interactive_terrain():
    print("Welcome to Interactive Terrain Generator!")

    water = float(input("Enter percentage of water (e.g., 0.4 for 40%): "))
    land = 1 - water
    print(f"Selected Water: {water * 100:.2f}%, Land: {land * 100:.2f}%")

    sand = float(input("Enter percentage of sand within land (e.g., 0.1 for 10%): "))
    green = float(input("Enter percentage of greenery within land (e.g., 0.6 for 60%): "))
    ice = 1 - (sand + green)
    print(f"Sand: {sand * 100:.2f}%, Greenery: {green * 100:.2f}%, Ice: {ice * 100:.2f}%")

    width, height = 100, 100
    perlin = perlin_2d(width, height, scale=50)
    worley = worley_2d(width, height)
    terrain = (0.6 * perlin + 0.4 * worley)
    terrain = simulate_erosion(terrain)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    biomes = classify_biomes(terrain, water_level=water, sand_width=sand * land, green_max=water + green * land)

    print("Displaying 2D Biome Map...")
    visualize_terrain(terrain, biomes, water, view_3d=False)

    print("Displaying 3D Terrain Map...")
    visualize_terrain(terrain, biomes, water, view_3d=True)

if __name__ == "__main__":
    interactive_terrain()
