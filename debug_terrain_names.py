import isaaclab.terrains as terrains
print("Attributes in isaaclab.terrains:")
for attr in dir(terrains):
    if "TerrainCfg" in attr:
        print(attr)
