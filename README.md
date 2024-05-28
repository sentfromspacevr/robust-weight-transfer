# Robust Weight Transfer for Blender

A Blender addon for getting smooth weight transfer results with just one click! 

Robust Weight Transfer can successfully transfer weights from the body, without needing additional weight painting work like smoothing in problem areas (between legs, between chest, armpits).

Weight transfer code is based on https://github.com/rin-23/RobustSkinWeightsTransferCode/tree/main

You can find a tutorial on the Gumroad page of the addon https://sentfromspacevr.gumroad.com/l/robust-weight-transfer

## Development

### Installing Dependencies

#### Option 1
When using the same Python version that Blender uses, you can install the dependencies with:
```
pip install --no-deps --target deps -r requirements.txt
```
#### Option 2
With this `pip` command you can control which platform and for which Python version wheels should be downloaded:
```
pip download --platform win_amd64 --python-version 310 --only-binary=:all: --no-deps -d whl -r requirements.txt
```
The dependency wheels get downloaded into the `whl` directory. From here you can unzip the content of the wheels into the `deps` directory

### VS Code Blender Extension

I recommend using https://github.com/JacquesLucke/blender_vscode during development.

## Academic Work

This Blender addon is based on "Robust Skin Weights Transfer via Weight Inpainting" by Rinat Abdrashitov, Kim Raichstat, Jared Monsen and David Hill published at SIGGRAPH ASIA 2023

You can find the project page here https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/index.html

When using the addon for academic work please cite them!

Changes to the original work:
- Allow flipped normals 
    - Handles solid meshes when vertex normals show into opposite direction of the vertex normals on the body
- Remesh using Robust Laplacian's point cloud Laplacian (During Point mode)
    - Workarounds `libigl.min_quad_with_fixed` failing during Inpaint when a disconnected mesh doesn't have any known values/matches



