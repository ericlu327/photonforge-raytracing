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
@group(0) @binding(1) var accum_in  : texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var accum_out : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var accum_sample : texture_2d<f32>;
@group(0) @binding(4) var samp : sampler;

fn rand(hash: vec2<u32>) -> f32 {
  var x = hash.x * 1664525u + 1013904223u + hash.y * 747796405u;
  x = (x ^ (x >> 16u)) * 2246822519u;
  x ^= (x >> 13u);
  let f = f32(x & 0x00FFFFFFu) / f32(0x01000000u);
  return clamp(f, 0.0, 0.999999);
}

struct Hit { dist: f32, n: vec3<f32>, albedo: vec3<f32>, emissive: vec3<f32>, mirror: f32 }

fn sphere_hit(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32,
              albedo: vec3<f32>, emissive: vec3<f32>, mirror: f32) -> Hit {
  let oc = ro - center;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - radius*radius;
  let h = b*b - c;
  if (h < 0.0) { return Hit(1e30, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0); }
  let t = -b - sqrt(h);
  if (t < 1e-3) { return Hit(1e30, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0); }
  let p = ro + rd * t;
  let n = normalize(p - center);
  return Hit(t, n, albedo, emissive, mirror);
}

fn plane_hit(ro: vec3<f32>, rd: vec3<f32>, y: f32, albedo: vec3<f32>) -> Hit {
  if (abs(rd.y) < 1e-4) { return Hit(1e30, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0); }
  let t = (y - ro.y) / rd.y;
  if (t < 1e-3) { return Hit(1e30, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0); }
  let n = vec3<f32>(0.0, 1.0, 0.0);
  return Hit(t, n, albedo, vec3<f32>(0.0), 0.0);
}

fn sky(rd: vec3<f32>) -> vec3<f32> {
  let t = 0.5 * (rd.y + 1.0);
  return mix(vec3<f32>(0.8, 0.9, 1.0), vec3<f32>(0.2, 0.3, 0.6), t);
}

fn onb(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(0.0,1.0,0.0), vec3<f32>(1.0,0.0,0.0), abs(n.y) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

fn cosine_sample_hemisphere(u: f32, v: f32) -> vec3<f32> {
  let r = sqrt(u);
  let theta = 6.2831853 * v;
  let x = r * cos(theta);
  let y = r * sin(theta);
  let z = sqrt(max(0.0, 1.0 - u));
  return vec3<f32>(x, y, z);
}
fn refract_ray(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
  let cosi = clamp(dot(-v, n), -1.0, 1.0);
  let cost2 = 1.0 - eta*eta*(1.0 - cosi*cosi);
  if (cost2 < 0.0) { // total internal reflection
    return reflect(v, n);
  }
  return normalize(eta*v + (eta*cosi - sqrt(cost2))*n);
}

// Schlick approximation
fn schlick_fresnel(cos_theta: f32, ior: f32) -> f32 {
  let r0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
  return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= cam.img_size.x || gid.y >= cam.img_size.y) { return; }

  let px = vec2<f32>(f32(gid.x), f32(gid.y));
  let res = vec2<f32>(f32(cam.img_size.x), f32(cam.img_size.y));
  let uv = (px + vec2<f32>(0.5, 0.5)) / res * 2.0 - 1.0;
  let aspect = res.x / max(1.0, res.y);

  // jitter per frame/pixel
  let jitter = vec2<f32>(
      rand(vec2<u32>(gid.x + cam.frame_index, gid.y)),
      rand(vec2<u32>(gid.y + cam.frame_index*3u, gid.x))
  );
  let jitter_uv = (jitter - 0.5) / res;

  // fixed 60° FOV
  let fov_tan = tan(0.5 * 60.0 * 0.0174532925);
  var rd = normalize(
      cam.dir +
      cam.right * (uv.x + jitter_uv.x) * aspect * fov_tan * 2.0 +
      cam.up    * (uv.y + jitter_uv.y)           * fov_tan * 2.0
  );
  var ro = cam.origin;

  var throughput = vec3<f32>(1.0);
  var radiance  = vec3<f32>(0.0);

  var bounce: u32 = 0u;
  loop {
    if (bounce > cam.max_bounce) { break; }

    // scene: two spheres + ground plane
    var best = Hit(1e30, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
    let s1 = sphere_hit(ro, rd, vec3<f32>(-1.2, 1.0, 0.0), 1.0, vec3<f32>(0.9, 0.25, 0.25), vec3<f32>(0.0), 0.0);
    if (s1.dist < best.dist) { best = s1; }
    let s2 = sphere_hit(ro, rd, vec3<f32>( 1.2, 1.0, 0.0), 1.0, vec3<f32>(0.25, 0.9, 0.3), vec3<f32>(0.0), 1.0); // mirror
    if (s2.dist < best.dist) { best = s2; }
    let pl = plane_hit(ro, rd, 0.0, vec3<f32>(0.8, 0.8, 0.8));
    if (pl.dist < best.dist) { best = pl; }

    if (best.dist == 1e30) {
      radiance += throughput * sky(rd);
      break;
    }

    let p = ro + rd * best.dist;
    let n = normalize(best.n);

    // simple fake point light
    let light_pos = vec3<f32>(0.0, 5.0, 2.5);
    let toL = light_pos - p;
    let distL = length(toL);
    let ldir = toL / max(1e-6, distL);
    let n_dot_l = max(0.0, dot(n, ldir));

    // hard shadow
    var shadowed = false;
    {
      let eps = 1e-3;
      var h = sphere_hit(p + n*eps, ldir, vec3<f32>(-1.2,1.0,0.0), 1.0, vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
      if (h.dist < distL) { shadowed = true; }
      h = sphere_hit(p + n*eps, ldir, vec3<f32>(1.2,1.0,0.0), 1.0, vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
      if (h.dist < distL) { shadowed = true; }
    }

    let light_color = vec3<f32>(8.0, 7.5, 7.0);
    if (!shadowed) {
      let falloff = 1.0 / (1.0 + 0.1*distL + 0.01*distL*distL);
      radiance += throughput * best.albedo * (light_color * n_dot_l * falloff);
    }

    // bounce
    let seed0 = rand(vec2<u32>(u32(p.x*4096.0) ^ u32(p.y*8192.0) ^ u32(p.z*16384.0) ^ cam.frame_index,
                               u32(gid.x + gid.y)));
    let seed1 = rand(vec2<u32>(u32(p.z*2048.0) ^ cam.frame_index,
                               u32(gid.y*3u + 7u)));

    if (best.mirror > 0.5) {
      // perfect reflection
      rd = reflect(rd, n);
      ro = p + n * 1e-3;
      throughput *= vec3<f32>(0.95);
    } else {
      // diffuse
      let TBN = onb(n);
      let local = cosine_sample_hemisphere(seed0, seed1);
      rd = normalize(TBN * local);
      ro = p + n * 1e-3;
      throughput *= best.albedo;
    }

    bounce += 1u;
  }
// progressive accumulation (use 2-arg textureLoad for storage textures)
let prev = textureLoad(accum_in, vec2<i32>(i32(gid.x), i32(gid.y)));
let f = f32(cam.frame_index);
let acc = (prev.rgb * f + radiance) / max(1.0, f + 1.0);
textureStore(accum_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(acc, 1.0));

}
