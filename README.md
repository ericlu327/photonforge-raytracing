### Ray-Traced Shadows & Reflections for Games
Real-time hybrid renderer: raster G-buffer -> DXR/Vulkan rays for shadows/reflections -> temporal+spatial denoise.

**Features**
- BVH (SAH, refit), PBR (GGX), blue-noise sampling
- Temporal accumulation, variance estimation, atrous filter
- ImGui HUD: ms per pass, spp, BVH stats

**Build**
- Windows: CMake + DX12/DXR + HLSL; Linux: Vulkan RT + GLSL
- Run: F5 to toggle RT, F6 to toggle denoise, `P` to save perf CSV

**Performance (1080p, RTX4070)**
- G-buffer: [x] ms | RT Shadows: [y] ms | RT Refl: [z] ms | Denoise: [w] ms | Total: [t] ms
