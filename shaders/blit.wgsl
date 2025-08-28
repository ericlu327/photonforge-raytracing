struct CameraUBO {
  origin     : vec3<f32>, _pad0 : f32,
  dir        : vec3<f32>, _pad1 : f32,
  right      : vec3<f32>, _pad2 : f32,
  up         : vec3<f32>, _pad3 : f32,
  img_size   : vec2<u32>,
  frame_index: u32,
  max_bounce : u32,
};

@group(0) @binding(0) var<uniform> cam : CameraUBO;
@group(0) @binding(3) var accum_tex : texture_2d<f32>;
@group(0) @binding(4) var samp : sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  let p = pos[vi];
  return vec4<f32>(p, 0.0, 1.0);
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x*(a*x+b)) / (x*(c*x+d)+e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / vec2<f32>(f32(cam.img_size.x), f32(cam.img_size.y));
  var color = textureSampleLevel(accum_tex, samp, uv, 0.0).rgb;
  color = aces_tonemap(color);
  color = pow(color, vec3<f32>(1.0/2.2));
  return vec4<f32>(color, 1.0);
}
