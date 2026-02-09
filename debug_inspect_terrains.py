
import isaaclab.terrains as terrain_gen
print("Available attributes in isaaclab.terrains:")
for attr in dir(terrain_gen):
    if "Terrain" in attr:
        print(attr)
