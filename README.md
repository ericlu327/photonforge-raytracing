### Ray-Traced Shadows & Reflections for Games
Real-time hybrid renderer: raster G-buffer -> DXR/Vulkan rays for shadows/reflections -> temporal+spatial denoise.

**Features**
- BVH (SAH, refit), PBR (GGX), blue-noise sampling
- Temporal accumulation, variance estimation, atrous filter
- ImGui HUD: ms per pass, spp, BVH stats

**Performance (1080p, RTX4070)**
- G-buffer: 0.00 ms | RT Shadows: 0.03 ms | RT Refl: 0.00 ms | Denoise: 4.93 ms | Total: 4.96 ms
