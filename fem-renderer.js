"use strict";

// =============================================================================
//  webgl-renderer.js
//
//  Two parts:
//
//  1) LIBRARY — matrix helpers, shader sources, program/VAO factories, and
//     geometry packing. Nothing in this part touches `gl` at load time, so
//     the file can be included on pages that create their own GL context
//     later (e.g. edge-editor.html).
//
//  2) STANDALONE VIEWER — the original turntable app. It consumes the four
//     globals produced by shape-builder.js (SURF_VERTS, SURF_INDICES,
//     LINE_VERTS, LINE_INDICES, plus the optional CONTOUR_* pair), packs
//     them into shared buffers, and drives its own render loop. Pages that
//     only want the library opt out by defining
//
//         const RENDERER_NO_AUTOSTART = true;
//
//     BEFORE this script is loaded.
// =============================================================================

// ---------------------------------------------------------------------------
//  Shared interaction / viewport state. Host pages read and write these
//  directly (mouse handlers update the turntable, resize() updates the
//  viewport metrics). Do NOT redeclare them in host scripts.
// ---------------------------------------------------------------------------
let turntableAngle  = 0;
let isDragging      = false;
let angularVelocity = 0;
let lastMouseX      = 0;

const FRICTION         = 0.95;
const DRAG_SENSITIVITY = 0.01;
const IDLE_SPIN        = 0.00;   // rad/frame when not dragging

// Node-editing state shared with the host page: `highlightNode` is the index
// (into the builder's NODES table) of the hovered/dragged marker, `editHold`
// pauses the idle spin while a node is being dragged, and the `current*`
// camera snapshot is written every frame so the host can pick and drag nodes
// in screen space.
let highlightNode = -1;
let editHold      = false;
let currentMvp = null, currentRot = null, currentProj = [1, 1];

// Assigned by the standalone viewer; host pages call it after editing nodes
// and rebuilding the builder's scene arrays. (Declared here because the
// viewer body is a strict-mode block, so its declarations aren't global.)
let refreshSceneBuffers = () => {};

let vp_w = 1, vp_h = 1, dpr = 1;

// ---------------------------------------------------------------------------
//  Matrix helpers (column-major)
// ---------------------------------------------------------------------------

function mat4_ortho(proj_w, proj_h, z_scale) {
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

// ---------------------------------------------------------------------------
//  Shader sources
// ---------------------------------------------------------------------------

const SURFACE_VS = `#version 300 es
  in vec3 v_position;
  in vec3 v_normal;
  in float v_value;   // scalar nodal field, basis-interpolated by the builder
  uniform mat4 m_mvp;
  uniform mat3 m_rot;
  out vec3 n_dir;
  out vec3 model_pos;
  out float field;
  void main() {
    n_dir = m_rot * v_normal;
    model_pos = v_position;
    field = v_value;
    gl_Position = m_mvp * vec4(v_position, 1.0);
  }
`;

const SURFACE_FS = `#version 300 es
  precision mediump float;
  in vec3 n_dir;
  in vec3 model_pos;
  in float field;
  uniform vec4 color;
  out vec4 outColor;
// Viridis colormap (polynomial fit, t ∈ [0,1]: deep violet → teal → yellow)
vec3 viridis(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = vec3(+0.277460, +0.002125, +0.309028);
  c += t * (vec3( -0.094373, +1.553700,  +2.512771)
     + t * (vec3( +1.982950, -1.607827, -11.676054)
     + t * (vec3(-14.333019, +2.088099, +28.586120)
     + t * (vec3(+24.806372, -1.129279, -33.551447)
     + t *  vec3(-11.654366, -0.005995, +13.928704)))));
  return clamp(c, 0.0, 1.0);
}
// Magma colormap (polynomial fit, t ∈ [0,1]: black → purple → orange → pale yellow)
vec3 magma(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = vec3(-0.016071, +0.008987, +0.001125);
  c += t * (vec3( +0.962977, +0.210677,  +2.081849)
     + t * (vec3( +0.536127, +1.469217,  +3.883827)
     + t * (vec3( +4.924098, -6.408635, -26.326736)
     + t * (vec3(-10.276817, +11.140276, +34.632792)
     + t *  vec3( +4.854209, -5.436745, -13.521122)))));
  return clamp(c, 0.0, 1.0);
}

// Plasma colormap (polynomial fit, t ∈ [0,1]: deep blue → magenta → orange → yellow)
vec3 plasma(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = vec3(+0.064169, +0.045211, +0.518986);
  c += t * (vec3(+2.003371, -0.731338, +1.479374)
     + t * (vec3(-1.132662, +2.693455, -4.610782)
     + t * (vec3(-0.153934, +0.985694, +3.262733)
     + t * (vec3(+0.888148, -4.587743, -0.557045)
     + t *  vec3(-0.731001, +2.581500, +0.045640)))));
  return clamp(c, 0.0, 1.0);
}
  // Classic FEM rainbow ("jet"). The quadratic field can over/undershoot
  // its nodal range between nodes, so clamp before mapping.
  vec3 jet(float t) {
    return clamp(1.5 - abs(4.0 * vec3(t) - vec3(3.0, 2.0, 1.0)), 0.0, 1.0);
  }
  void main() {
    //vec3 c = mix(jet(clamp(field, 0.0, 1.0)), vec3(1.0), 0.12);
    vec3 c = mix(magma(field), vec3(1.0), 0.8);
    vec3 l = normalize(vec3(0.35, 0.5, 0.8));
    c *= (0.55 + 0.45 * max(0.0, dot(n_dir, l)));
    outColor = vec4(c * color.a, color.a);
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

// FEM node markers as round GL points: solid = corner node, hollow =
// mid-edge node, orange = hovered/dragged. v_flag encodes 0|1 corner|mid,
// +2 when highlighted.
const NODE_VS = `#version 300 es
  in vec3 v_position;
  in float v_flag;
  uniform mat4 m_mvp;
  uniform float point_scale;
  out float flag;
  void main() {
    flag = v_flag;
    gl_Position = m_mvp * vec4(v_position, 1.0);
    gl_Position.z -= 0.012;   // in front of surface and edge lines
    gl_PointSize = (v_flag > 1.5 ? 16.0 : 11.0) * point_scale;
  }
`;

const NODE_FS = `#version 300 es
  precision mediump float;
  in float flag;
  out vec4 outColor;
  void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    vec3 ink = vec3(0.24, 0.26, 0.30);
    vec3 fill = flag > 1.5          ? vec3(1.0, 0.45, 0.15)
              : mod(flag, 2.0) > 0.5 ? vec3(1.0)
                                     : ink;
    outColor = vec4(r2 > 0.5 ? ink : fill, 1.0);
  }
`;

const CONTOUR_VS = `#version 300 es
  in vec3 v_position;
  in vec3 v_dir;    // outward end direction (format-2 side encoding)
  in vec3 v_na;     // face normal A
  in vec3 v_nb;     // face normal B
  uniform mat4 m_mvp;
  uniform mat3 m_rot;
  uniform vec4 line_p;
  void main() {
    // Silhouette test (orthographic, view dir = +z): draw only when the two
    // adjacent faces straddle the view plane.
    float fa = (m_rot * v_na).z;
    float fb = (m_rot * v_nb).z;
    if (fa * fb > 0.0) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);   // collapse: zero-area quad
      return;
    }
    vec3 e = v_dir;
    float perp_sign = -1.0;
    if (dot(e, e) > 1.5) { perp_sign = 1.0; e *= 0.5; }
    perp_sign *= line_p.w;
    vec4 position = m_mvp * vec4(v_position + e * line_p.x, 1.0);
    vec3 se = (m_mvp * vec4(e, 0.0)).xyz;
    vec2 v  = vec2(se.x / line_p.z, se.y);
    float l = length(v);
    vec2 dir_px  = l > 1e-6 ? v / l : vec2(1.0, 0.0);
    vec2 perp_px = vec2(-dir_px.y, dir_px.x);
    position.x += line_p.x * line_p.z * perp_px.x * perp_sign;
    position.y += line_p.x *            perp_px.y * perp_sign;
    position.z -= 0.008;   // stronger bias: silhouettes lie on grazing surfaces
    gl_Position = position;
  }
`;

// ---------------------------------------------------------------------------
//  Program factories
// ---------------------------------------------------------------------------

const SURFACE_ATTR = { v_position: 0, v_normal: 1, v_value: 2 };
const CONTOUR_ATTR = { v_position: 0, v_dir: 1, v_na: 2, v_nb: 3 };
const NODE_ATTR    = { v_position: 0, v_flag: 1 };

function compileShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s) + "\n" + src);
  return s;
}

function linkProgram(gl, vsSrc, fsSrc, attrs) {
  const vs = compileShader(gl, gl.VERTEX_SHADER,   vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
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

// Builds the full standard program set. Pages that don't need all four can
// simply ignore the ones they don't use.
function createRenderPrograms(gl) {
  return {
    surface: linkProgram(gl, SURFACE_VS, SURFACE_FS, SURFACE_ATTR),
    cross:   linkProgram(gl, SURFACE_VS, CROSS_FS,   SURFACE_ATTR),
    line:    linkProgram(gl, LINE_VS,    LINE_FS,    SURFACE_ATTR),
    contour: linkProgram(gl, CONTOUR_VS, LINE_FS,    CONTOUR_ATTR),
  };
}

// ---------------------------------------------------------------------------
//  VAO factories
// ---------------------------------------------------------------------------

// Surface + crease-line layout: 28-byte stride [pos(3), normal(3), value(1)].
// The trailing field value is consumed by the surface program only; the line
// program simply ignores the enabled-but-unused attribute.
function createSurfaceVao(gl, vbo, ibo) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.enableVertexAttribArray(SURFACE_ATTR.v_position);
  gl.vertexAttribPointer(SURFACE_ATTR.v_position, 3, gl.FLOAT, false, 28, 0);
  gl.enableVertexAttribArray(SURFACE_ATTR.v_normal);
  gl.vertexAttribPointer(SURFACE_ATTR.v_normal,   3, gl.FLOAT, false, 28, 12);
  gl.enableVertexAttribArray(SURFACE_ATTR.v_value);
  gl.vertexAttribPointer(SURFACE_ATTR.v_value,    1, gl.FLOAT, false, 28, 24);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bindVertexArray(null);
  return vao;
}

// Contour (silhouette) candidate layout: 48-byte stride
// [pos(3), dir(3), face normal A(3), face normal B(3)].
function createContourVao(gl, cvbo, cibo) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, cvbo);
  gl.enableVertexAttribArray(CONTOUR_ATTR.v_position);
  gl.vertexAttribPointer(CONTOUR_ATTR.v_position, 3, gl.FLOAT, false, 48, 0);
  gl.enableVertexAttribArray(CONTOUR_ATTR.v_dir);
  gl.vertexAttribPointer(CONTOUR_ATTR.v_dir,      3, gl.FLOAT, false, 48, 12);
  gl.enableVertexAttribArray(CONTOUR_ATTR.v_na);
  gl.vertexAttribPointer(CONTOUR_ATTR.v_na,       3, gl.FLOAT, false, 48, 24);
  gl.enableVertexAttribArray(CONTOUR_ATTR.v_nb);
  gl.vertexAttribPointer(CONTOUR_ATTR.v_nb,       3, gl.FLOAT, false, 48, 36);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cibo);
  gl.bindVertexArray(null);
  return vao;
}

// ---------------------------------------------------------------------------
//  Geometry packing
//
//  Packs surface + crease-line data into a single shared VBO/IBO payload:
//  surface vertices come first, line vertices second, line indices are
//  offset by the surface vertex count so they reference the right slice of
//  the shared buffer.
// ---------------------------------------------------------------------------

function packSurfaceAndLines(surfVerts, surfIndices, lineVerts, lineIndices) {
  const surfVertCount = surfVerts.length / 7;

  const verts = new Float32Array(surfVerts.length + lineVerts.length);
  verts.set(surfVerts, 0);
  verts.set(lineVerts, surfVerts.length);

  const indices = new Uint32Array(surfIndices.length + lineIndices.length);
  indices.set(surfIndices, 0);
  for (let i = 0; i < lineIndices.length; i++)
    indices[surfIndices.length + i] = lineIndices[i] + surfVertCount;

  return {
    verts,
    indices,
    surfIndexOffset: 0,
    surfIndexCount:  surfIndices.length,
    lineIndexOffset: surfIndices.length,
    lineIndexCount:  lineIndices.length,
  };
}

// line_p uniform for the line/contour shaders: [half-width in clip units,
// end-cap extension, aspect correction, perpendicular sign]. Reads the
// shared vp_w / vp_h / dpr globals, so call it per frame after resize().
function makeLineParams(widthPx, sign) {
  return new Float32Array([
    (widthPx * dpr) / vp_h,
    0.01,
    vp_h / vp_w,
    sign === undefined ? 1.0 : sign,
  ]);
}

// =============================================================================
//  STANDALONE VIEWER — skipped when RENDERER_NO_AUTOSTART is set.
//
//  Expects the host page to provide `gl` and `resize()`, and shape-builder.js
//  to provide SURF_VERTS / SURF_INDICES / LINE_VERTS / LINE_INDICES (plus,
//  optionally, CONTOUR_VERTS / CONTOUR_INDICES).
// =============================================================================

if (typeof RENDERER_NO_AUTOSTART === "undefined" || !RENDERER_NO_AUTOSTART) {

  const packed = packSurfaceAndLines(SURF_VERTS, SURF_INDICES,
                                     LINE_VERTS, LINE_INDICES);

  const models = {
    geometry: {
      index_offset:      packed.surfIndexOffset,
      index_count:       packed.surfIndexCount,
      line_index_offset: packed.lineIndexOffset,
      line_index_count:  packed.lineIndexCount,
    },
  };

  const programs      = createRenderPrograms(gl);
  const surfaceShader = programs.surface;
  const crossShader   = programs.cross;
  const lineShader    = programs.line;
  const contourShader = programs.contour;

  // ---------------------------------------------------------------------------
  //  Buffers + single shared VAO
  // ---------------------------------------------------------------------------

  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, packed.verts, gl.DYNAMIC_DRAW);

  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, packed.indices, gl.STATIC_DRAW);

  const vao = createSurfaceVao(gl, vbo, ibo);

  // Re-upload vertex data after the host edits nodes and calls the builder's
  // rebuildScene(). Topology is fixed (same tessellation, same index buffer),
  // so only the shared VBO changes.
  refreshSceneBuffers = function () {
    const p = packSurfaceAndLines(SURF_VERTS, SURF_INDICES,
                                  LINE_VERTS, LINE_INDICES);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, p.verts, gl.DYNAMIC_DRAW);
  };

  // Node markers (editable control points): a small dynamic point buffer,
  // refilled every frame from the builder's live node positions. Guarded so
  // the viewer still works with a builder that has no editable nodes.
  const HAS_NODES = typeof NODES !== "undefined" && NODES.length > 0;
  let nodeProg = null, nodeVbo = null, nodeVao = null, nodeData = null;
  if (HAS_NODES) {
    nodeProg = linkProgram(gl, NODE_VS, NODE_FS, NODE_ATTR);
    nodeVbo  = gl.createBuffer();
    nodeVao  = gl.createVertexArray();
    gl.bindVertexArray(nodeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeVbo);
    gl.enableVertexAttribArray(NODE_ATTR.v_position);
    gl.vertexAttribPointer(NODE_ATTR.v_position, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(NODE_ATTR.v_flag);
    gl.vertexAttribPointer(NODE_ATTR.v_flag,     1, gl.FLOAT, false, 16, 12);
    gl.bindVertexArray(null);
    nodeData = new Float32Array(NODES.length * 4);
  }

  function drawNodes(mvp) {
    for (let k = 0; k < NODES.length; k++) {
      const p = nodeWorldPos(k);
      nodeData[k*4]   = p[0];
      nodeData[k*4+1] = p[1];
      nodeData[k*4+2] = p[2];
      nodeData[k*4+3] = (NODES[k].mid ? 1 : 0) + (k === highlightNode ? 2 : 0);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeVbo);
    gl.bufferData(gl.ARRAY_BUFFER, nodeData, gl.DYNAMIC_DRAW);
    gl.useProgram(nodeProg.program);
    gl.bindVertexArray(nodeVao);
    gl.uniformMatrix4fv(nodeProg.uniforms["m_mvp"], false, mvp);
    gl.uniform1f(nodeProg.uniforms["point_scale"], dpr);
    gl.drawArrays(gl.POINTS, 0, NODES.length);
    gl.bindVertexArray(null);
  }

  // Contour (silhouette) candidates: separate buffers, 48-byte stride.
  // Guarded so this renderer still works with a builder that predates them.
  const HAS_CONTOURS = typeof CONTOUR_VERTS !== "undefined" &&
                       CONTOUR_VERTS.length > 0;
  let contourVao = null, contourIndexCount = 0;
  if (HAS_CONTOURS) {
    const cvbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cvbo);
    gl.bufferData(gl.ARRAY_BUFFER, CONTOUR_VERTS, gl.STATIC_DRAW);
    const cibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, CONTOUR_INDICES, gl.STATIC_DRAW);
    contourVao = createContourVao(gl, cvbo, cibo);
    contourIndexCount = CONTOUR_INDICES.length;
  }

  // ---------------------------------------------------------------------------
  //  draw_mesh — same as v3
  // ---------------------------------------------------------------------------

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

    const lineP = makeLineParams(1.0, backface ? -1.0 : 1.0);   // thick outlines
    const lineCol = new Float32Array([col[0]*0.4, col[1]*0.4, col[2]*0.4, col[3]]);

    gl.useProgram(lineShader.program);
    gl.uniformMatrix4fv(lineShader.uniforms["m_mvp"],  false, mvp);
    gl.uniform4fv(      lineShader.uniforms["line_p"],         lineP);
    gl.uniform4fv(      lineShader.uniforms["color"],          lineCol);
    gl.drawElements(gl.TRIANGLES, mesh.line_index_count,
                    gl.UNSIGNED_INT, mesh.line_index_offset * 4);

    // View-dependent silhouette outlines on smooth surfaces (cylinder walls,
    // sphere rims): the shader draws only candidates that straddle the view.
    if (HAS_CONTOURS) {
      gl.useProgram(contourShader.program);
      gl.bindVertexArray(contourVao);
      gl.uniformMatrix4fv(contourShader.uniforms["m_mvp"], false, mvp);
      gl.uniformMatrix3fv(contourShader.uniforms["m_rot"], false,
                          mat3_from_mat4(rotation));
      gl.uniform4fv(contourShader.uniforms["line_p"], lineP);
      gl.uniform4fv(contourShader.uniforms["color"],  lineCol);
      gl.drawElements(gl.TRIANGLES, contourIndexCount, gl.UNSIGNED_INT, 0);
    }

    gl.bindVertexArray(null);
  }

  gl.clearColor(1.0, 1.0, 1.0, 1.0);
  gl.clearDepth(1.0);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);

  const GEOMETRY_COLOR = [0.890, 0.871, 0.796];

  function frame() {
    resize();
    if (!isDragging && !editHold) {
      turntableAngle  += angularVelocity + IDLE_SPIN;
      angularVelocity *= FRICTION;
    }

    const tilt = mat4_rotX(0.45);           // fixed forward tilt
    const spin = mat4_rotY(turntableAngle);
    const rot  = mat4_mul(tilt, spin);

    const aspect = vp_w / vp_h;
    const proj_h = Math.max(1.55, 2.2 / aspect);   // keep both elements framed
    const proj_w = proj_h * aspect;
    const proj = mat4_ortho(proj_w, proj_h, -0.4);
    const mvp = mat4_mul(proj, rot);

    currentMvp  = mvp;                      // camera snapshot for host picking
    currentRot  = rot;
    currentProj = [proj_w, proj_h];

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    draw_mesh("geometry", mvp, rot, GEOMETRY_COLOR, 1.0, false, false, false);
    if (HAS_NODES) drawNodes(mvp);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

}
