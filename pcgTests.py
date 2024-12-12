import unittest
import numpy as np
from pcg import perlin_2d, worley_2d, simulate_erosion, classify_biomes

class TestTerrainGeneration(unittest.TestCase):
    def setUp(self):
        #set a fixed random seed 
        np.random.seed(42)

    def test_perlin_noise_shape(self):
        #test that Perlin noise generates correct shape
        noise = perlin_2d(50, 50)
        self.assertEqual(noise.shape, (50, 50))
        self.assertTrue(np.all(noise >= 0) and np.all(noise <= 1))

    def test_perlin_noise_values(self):
        #test specific properties of Perlin noise
        noise = perlin_2d(50, 50, scale=50)
        self.assertAlmostEqual(noise.min(), 0, places=7)
        self.assertAlmostEqual(noise.max(), 1, places=7)

    def test_worley_noise_shape(self):
        #test Worley noise generation
        noise = worley_2d(50, 50, num_points=10)
        self.assertEqual(noise.shape, (50, 50))
        self.assertTrue(np.all(noise >= 0) and np.all(noise <= 1))

    def test_erosion_simulation(self):
        #test erosion simulation maintains value range
        base_terrain = np.random.rand(50, 50)
        eroded_terrain = simulate_erosion(base_terrain)
        self.assertTrue(np.all(eroded_terrain >= 0) and np.all(eroded_terrain <= 1))

    def test_biome_classification(self):
        #test biome classification logic
        terrain = np.linspace(0, 1, 100).reshape(10, 10)
        biomes = classify_biomes(terrain)
        #check water classification
        self.assertTrue(np.all(biomes[terrain <= 0.4] <= 0.4))
        #check ice classification
        self.assertTrue(np.all(biomes[terrain > 0.8] == 4))

    def test_terrain_generation_integration(self):
        #test full terrain generation process
        width, height = 100, 100
        perlin = perlin_2d(width, height)
        worley = worley_2d(width, height)
        combined_terrain = (0.6 * perlin + 0.4 * worley)
        eroded_terrain = simulate_erosion(combined_terrain)
        
        #verify terrain properties
        self.assertEqual(eroded_terrain.shape, (width, height))
        self.assertTrue(np.all(eroded_terrain >= 0) and np.all(eroded_terrain <= 1))

if __name__ == '__main__':
    # python -m unittest pcgTests.py <- to run tests, make sure pcg.py and pcgTests.py is in the same folder!
    unittest.main()