"use strict";

let turntableAngle = 0;
let isDragging = false;
let angularVelocity = 0;
let lastMouseX = 0;

const FRICTION = 0.7;
const DRAG_SENSITIVITY = 0.01;

function mat4_ortho_simple(proj_w, proj_h, z_scale) {
  return new Float32Array([
    1 / proj_w, 0,          0,        0,
    0,          1 / proj_h, 0,        0,
    0,          0,          z_scale,  0,
    0,          0,          0,        1,
  ]);
}

function mat4_mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k*4 + r] * b[c*4 + k];
      o[c*4 + r] = s;
    }
  return o;
}

function mat4_rotX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([
    1, 0, 0, 0,
    0, c, s, 0,
    0,-s, c, 0,
    0, 0, 0, 1,
  ]);
}

function mat4_rotY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([
    c, 0, s, 0,
    0, 1, 0, 0,
   -s, 0, c, 0,
    0, 0, 0, 1,
  ]);
}

function mat4_rotZ(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([
     c, s, 0, 0,
    -s, c, 0, 0,
     0, 0, 1, 0,
     0, 0, 0, 1,
  ]);
}

function mat3_from_mat4(m) {
  return new Float32Array([m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]]);
}

// =============================================================================
//  Pack the preprocessed data into a single shared VBO + IBO.
//
//  Same layout as the torus version: surface vertices come first, line
//  vertices second, line indices are offset by the surface vertex count so
//  they reference the right slice of the shared buffer.
// =============================================================================

const surfaceVertCount = SURF_VERTS.length / 6;
const lineVertCount    = LINE_VERTS.length / 6;

const allVerts = new Float32Array(SURF_VERTS.length + LINE_VERTS.length);
allVerts.set(SURF_VERTS, 0);
allVerts.set(LINE_VERTS, SURF_VERTS.length);

const offsetLineIdx = new Uint32Array(LINE_INDICES.length);
for (let i = 0; i < LINE_INDICES.length; i++)
  offsetLineIdx[i] = LINE_INDICES[i] + surfaceVertCount;

const allIndices = new Uint32Array(SURF_INDICES.length + offsetLineIdx.length);
allIndices.set(SURF_INDICES, 0);
allIndices.set(offsetLineIdx, SURF_INDICES.length);

const models = {
  geometry: {
    index_offset:      0,
    index_count:       SURF_INDICES.length,
    line_index_offset: SURF_INDICES.length,
    line_index_count:  offsetLineIdx.length,
  },
};

const SURFACE_VS = `#version 300 es
  in vec3 v_position;
  in vec3 v_normal;
  uniform mat4 m_mvp;
  uniform mat3 m_rot;
  out vec3 n_dir;
  out vec3 model_pos;
  void main() {
    n_dir = m_rot * v_normal;
    model_pos = v_position;
    gl_Position = m_mvp * vec4(v_position, 1.0);
  }
`;

const SURFACE_FS = `#version 300 es
  precision mediump float;
  in vec3 n_dir;
  in vec3 model_pos;
  uniform vec4 color;
  out vec4 outColor;
  void main() {
    vec4 c = color;
    c.rgb *= (0.75 + 0.25 * max(0.0, n_dir.z));
    outColor = c;
  }
`;

const CROSS_FS = `#version 300 es
  precision mediump float;
  in vec3 n_dir;
  in vec3 model_pos;
  uniform vec4 color;
  uniform vec4 cross_section_plane;
  uniform vec3 cross_section_param;
  out vec4 outColor;
  void main() {
    vec4 c = color;
    if (dot(model_pos, cross_section_plane.xyz) > cross_section_plane.w) {
      float t = dot(model_pos, cross_section_param);
      c.rgb *= 0.85 + 0.2 * clamp(5.0 * sin(t * 10.0), 0.0, 1.0);
    }
    c.rgb *= (0.75 + 0.25 * max(0.0, n_dir.z));
    outColor = c;
  }
`;

const LINE_VS = `#version 300 es
  in vec3 v_position;
  in vec3 v_normal;   // outward END DIRECTION of the segment (-d at p0, +d at
                      // p1); doubled length (dot > 1.5) marks the other quad
                      // side. Requires line data built by shape-builder.js.
  uniform mat4 m_mvp;
  uniform vec4 line_p;
  void main() {
    vec3 e = v_normal;
    float perp_sign = -1.0;
    if (dot(e, e) > 1.5) {
      perp_sign = 1.0;
      e *= 0.5;
    }
    perp_sign *= line_p.w;

    // Square end cap: extend the segment slightly past its endpoint so
    // adjacent crease lines meet cleanly at corners.
    vec4 position = m_mvp * vec4(v_position + e * line_p.x, 1.0);

    // Project the edge direction to aspect-corrected (pixel-ish) space and
    // extrude perpendicular to IT, not to the surface normal: this makes the
    // stroke width identical for every edge, at every camera angle.
    vec3 se = (m_mvp * vec4(e, 0.0)).xyz;
    vec2 v  = vec2(se.x / line_p.z, se.y);
    float l = length(v);
    vec2 dir_px  = l > 1e-6 ? v / l : vec2(1.0, 0.0);
    vec2 perp_px = vec2(-dir_px.y, dir_px.x);

    float width = line_p.x;
    position.x += width * line_p.z * perp_px.x * perp_sign;
    position.y += width *            perp_px.y * perp_sign;
    position.z -= 0.003;   // depth bias replaces the old normal-push,
                           // keeping the line in front of the surface
    gl_Position = position;
  }
`;
const LINE_FS = `#version 300 es
  precision mediump float;
  uniform vec4 color;
  out vec4 outColor;
  void main() { outColor = color; }
`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s) + "\n" + src);
  return s;
}

function link(vsSrc, fsSrc, attrs) {
  const vs = compile(gl.VERTEX_SHADER,   vsSrc);
  const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs); gl.attachShader(p, fs);
  for (const [name, loc] of Object.entries(attrs))
    gl.bindAttribLocation(p, loc, name);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  gl.deleteShader(vs); gl.deleteShader(fs);
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    u[info.name] = gl.getUniformLocation(p, info.name);
  }
  return { program: p, uniforms: u };
}

const ATTR = { v_position: 0, v_normal: 1 };
const surfaceShader = link(SURFACE_VS, SURFACE_FS, ATTR);
const crossShader   = link(SURFACE_VS, CROSS_FS,   ATTR);
const lineShader    = link(LINE_VS,    LINE_FS,    ATTR);

// =============================================================================
//  Buffers + single shared VAO
// =============================================================================

const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, allVerts, gl.STATIC_DRAW);

const ibo = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, allIndices, gl.STATIC_DRAW);

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.enableVertexAttribArray(ATTR.v_position);
gl.vertexAttribPointer(ATTR.v_position, 3, gl.FLOAT, false, 24, 0);
gl.enableVertexAttribArray(ATTR.v_normal);
gl.vertexAttribPointer(ATTR.v_normal,   3, gl.FLOAT, false, 24, 12);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
gl.bindVertexArray(null);

// =============================================================================
//  draw_mesh — same as v3
// =============================================================================

let vp_w = 1, vp_h = 1, dpr = 1;

function draw_mesh(name, mvp, rotation, color, opacity, backface, cross_section, skip_line) {
  if (opacity === undefined)  opacity = 1.0;
  if (backface === undefined) backface = false;

  let plane = null, param = null;
  if (cross_section === true) {
    plane = [0, 1, 0, 0.01];
    param = [0.01, 0.01, 0.0];
  } else if (Array.isArray(cross_section)) {
    plane = cross_section[0];
    param = cross_section[1];
  }

  const mesh = models[name];

  if (opacity === 1.0) {
    gl.disable(gl.BLEND);
    gl.depthMask(true);
  } else {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
  }
  gl.enable(gl.CULL_FACE);
  gl.cullFace(backface ? gl.FRONT : gl.BACK);

  const col = new Float32Array([
    color[0] * opacity,
    color[1] * opacity,
    color[2] * opacity,
    opacity,
  ]);

  const surfProg = plane ? crossShader : surfaceShader;
  gl.useProgram(surfProg.program);
  gl.bindVertexArray(vao);
  gl.uniformMatrix4fv(surfProg.uniforms["m_mvp"], false, mvp);
  gl.uniformMatrix3fv(surfProg.uniforms["m_rot"], false, mat3_from_mat4(rotation));
  gl.uniform4fv(      surfProg.uniforms["color"], col);
  if (plane) {
    gl.uniform4fv(surfProg.uniforms["cross_section_plane"], plane);
    gl.uniform3fv(surfProg.uniforms["cross_section_param"], param);
  }
  gl.drawElements(gl.TRIANGLES, mesh.index_count,
                  gl.UNSIGNED_INT, mesh.index_offset * 4);

  if (skip_line) { gl.bindVertexArray(null); return; }

  const lineP = new Float32Array([
    (1.1 * 1.1 * dpr) / vp_h,
    0.01,
    vp_h / vp_w,
    backface ? -1.0 : 1.0,
  ]);
  const lineCol = new Float32Array([col[0]*0.6, col[1]*0.6, col[2]*0.6, col[3]]);

  gl.useProgram(lineShader.program);
  gl.uniformMatrix4fv(lineShader.uniforms["m_mvp"],  false, mvp);
  gl.uniform4fv(      lineShader.uniforms["line_p"],         lineP);
  gl.uniform4fv(      lineShader.uniforms["color"],          lineCol);
  gl.drawElements(gl.TRIANGLES, mesh.line_index_count,
                  gl.UNSIGNED_INT, mesh.line_index_offset * 4);

  gl.bindVertexArray(null);
}

gl.clearColor(1.0, 1.0, 1.0, 1.0);
gl.clearDepth(1.0);
gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);

const GEOMETRY_COLOR = [0.890, 0.871, 0.796];

const CS_PLANE = [0.0, 1.0, 0.0, +1.195];
const CS_PARAM = [10., 0.0, 10.];

const startTime = performance.now();

function frame(now) {
  resize();
  const t = (now - startTime) * 0.001;
  if (!isDragging) {
    turntableAngle  += angularVelocity * t;
    angularVelocity *= FRICTION;
  }

  const tilt = mat4_rotX(0.45);          // fixed forward tilt, unchanged
  const spin = mat4_rotY(turntableAngle); // ← was t * 0.18
  const rot  = mat4_mul(tilt, spin);

  const aspect = vp_w / vp_h;
  const proj_h = 1.5;
  const proj_w = proj_h * aspect;
  const proj = mat4_ortho_simple(proj_w, proj_h, -0.4);
  const mvp = mat4_mul(proj, rot);

  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  draw_mesh(
    "geometry",
    mvp, rot,
    GEOMETRY_COLOR,
    1.0,
    false,
    [CS_PLANE, CS_PARAM],
    false
  );

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
