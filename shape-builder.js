"use strict";

// =============================================================================
//  shape-builder.js
//
//  Defines composite 3D shapes (boxes, spheres, wedges, cylinders) in code and produces
//  the four globals that webgl-code.js expects:
//
//      SURF_VERTS    Float32Array   interleaved [px py pz nx ny nz] * N
//      SURF_INDICES  Uint32Array    triangle indices into SURF_VERTS
//      LINE_VERTS    Float32Array   same 6-float layout (pos + "extrusion" normal)
//      LINE_INDICES  Uint32Array    triangle indices for the crease-line quads
//
//  Load order in the HTML matters:
//
//      <script src="shape-builder.js"></script>   <!-- first -->
//      <script src="webgl-code.js"></script>      <!-- second -->
//
//  Soft vs. hard edges are decided automatically from the dihedral angle
//  between adjacent faces (angle >= creaseAngle -> hard: split normals + a
//  crease line; otherwise soft: smoothed normals, no line), and can then be
//  OVERRIDDEN per edge:
//
//      const mesh = builder.build({
//        creaseAngleDeg: 40,
//        overrides: [
//          // Force an edge hard (adds a crease line + splits shading):
//          { a: [-0.5, 0.17, -0.5], b: [0.5, 0.17, -0.5], hard: true },
//          // Force an edge soft (removes an incorrectly detected crease):
//          { a: [ 0.0, 0.83,  0.45], b: [0.07, 0.83, 0.44], hard: false },
//        ],
//        overrideEps: 1e-3,   // endpoint matching tolerance (model units)
//      });
//
//  Endpoints are given in final (transformed) model space; order of a/b does
//  not matter. Use edge-editor.html to pick edges interactively and export
//  this overrides array instead of typing coordinates by hand.
// =============================================================================

// ----------------------------------------------------------------------------
//  Small vector helpers (self-contained; this file loads before the renderer)
// ----------------------------------------------------------------------------

function sb_sub(a, b)   { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function sb_add(a, b)   { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function sb_scale(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }
function sb_cross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}
function sb_dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function sb_len(a)    { return Math.hypot(a[0], a[1], a[2]); }
function sb_norm(a) {
  const l = sb_len(a);
  return l > 1e-12 ? sb_scale(a, 1 / l) : [0, 0, 0];
}

// Builds a point-transform closure from {scale, rotate:[rx,ry,rz], translate}.
// Application order: scale -> rotX -> rotY -> rotZ -> translate.
function sb_xform(opts) {
  const t = (opts && opts.translate) || [0, 0, 0];
  const r = (opts && opts.rotate)    || [0, 0, 0];
  let   s = (opts && opts.scale) === undefined ? 1 : opts.scale;
  if (typeof s === "number") s = [s, s, s];

  const cx = Math.cos(r[0]), sx = Math.sin(r[0]);
  const cy = Math.cos(r[1]), sy = Math.sin(r[1]);
  const cz = Math.cos(r[2]), sz = Math.sin(r[2]);

  return function (p) {
    let x = p[0]*s[0], y = p[1]*s[1], z = p[2]*s[2];
    let y1 = cx*y - sx*z, z1 = sx*y + cx*z;         // rotX
    y = y1; z = z1;
    let x1 = cy*x + sy*z, z2 = -sy*x + cy*z;        // rotY
    x = x1; z = z2;
    let x2 = cz*x - sz*y, y2 = sz*x + cz*y;         // rotZ
    x = x2; y = y2;
    return [x + t[0], y + t[1], z + t[2]];
  };
}

// ----------------------------------------------------------------------------
//  ShapeBuilder — collect raw triangles from primitives, then build()
// ----------------------------------------------------------------------------

class ShapeBuilder {
  constructor() {
    this.tris = [];   // flat: 9 floats per triangle (3 positions)
  }

  // Canonical, order-insensitive key for an edge given its two endpoints in
  // final model space. Used to match user overrides against detected edges.
  static edgeKey(a, b, eps) {
    eps = eps || 1e-3;
    const q = (p) => Math.round(p[0]/eps) + "," +
                     Math.round(p[1]/eps) + "," +
                     Math.round(p[2]/eps);
    const ka = q(a), kb = q(b);
    return ka < kb ? ka + "|" + kb : kb + "|" + ka;
  }

  // Add raw triangles: points = [[x,y,z], ...], indices = [i0,i1,i2, ...]
  addTriangles(points, indices, opts) {
    const xf = sb_xform(opts);
    for (let i = 0; i < indices.length; i += 3) {
      const a = xf(points[indices[i]]);
      const b = xf(points[indices[i + 1]]);
      const c = xf(points[indices[i + 2]]);
      this.tris.push(a[0],a[1],a[2], b[0],b[1],b[2], c[0],c[1],c[2]);
    }
    return this;
  }

  // Axis-aligned box centered at origin (before transform).
  // opts: { size:[w,h,d], translate, rotate, scale }
  addBox(opts) {
    const size = (opts && opts.size) || [1, 1, 1];
    const w = size[0] / 2, h = size[1] / 2, d = size[2] / 2;
    const P = [
      [-w,-h,-d], [ w,-h,-d], [ w, h,-d], [-w, h,-d],   // back  (z = -d)
      [-w,-h, d], [ w,-h, d], [ w, h, d], [-w, h, d],   // front (z = +d)
    ];
    const I = [
      4,5,6, 4,6,7,   // front  (+z)
      1,0,3, 1,3,2,   // back   (-z)
      5,1,2, 5,2,6,   // right  (+x)
      0,4,7, 0,7,3,   // left   (-x)
      3,7,6, 3,6,2,   // top    (+y)
      0,1,5, 0,5,4,   // bottom (-y)
    ];
    return this.addTriangles(P, I, opts);
  }

  // Triangular prism ("wedge"): right-triangle cross-section in XY, extruded
  // along Z, centered at origin. Vertical face at -x, flat bottom at -y,
  // sloped (hypotenuse) face rising from +x to +y.
  addWedge(opts) {
    const size = (opts && opts.size) || [1, 1, 1];
    const w = size[0] / 2, h = size[1] / 2, d = size[2] / 2;
    const P = [
      [-w,-h, d], [ w,-h, d], [-w, h, d],   // 0,1,2  front triangle (z = +d)
      [-w,-h,-d], [ w,-h,-d], [-w, h,-d],   // 3,4,5  back  triangle (z = -d)
    ];
    const I = [
      0,1,2,          // front  (+z)
      3,5,4,          // back   (-z)
      0,3,4, 0,4,1,   // bottom (-y)
      0,2,5, 0,5,3,   // left   (-x)
      1,4,5, 1,5,2,   // slope  (+x,+y)
    ];
    return this.addTriangles(P, I, opts);
  }

  // Cylinder along the Y axis, centered at origin. The curved wall is
  // auto-detected as soft (smooth shading), the two rim circles as hard
  // (crease lines) — no overrides needed. Set radiusTop / radiusBottom
  // separately for truncated cones, or radiusTop: 0 for a cone (the
  // degenerate top collapses cleanly; the apex shades smoothly).
  // opts: { radius, radiusTop, radiusBottom, height, segments,
  //         translate, rotate, scale }
  addCylinder(opts) {
    opts = opts || {};
    const r        = opts.radius !== undefined ? opts.radius : 0.5;
    const rt       = opts.radiusTop    !== undefined ? opts.radiusTop    : r;
    const rb       = opts.radiusBottom !== undefined ? opts.radiusBottom : r;
    const h        = (opts.height   || 1) / 2;
    const segments = opts.segments || 32;

    const P = [];
    for (let i = 0; i < segments; i++) {           // 0..seg-1: top ring
      const phi = (i / segments) * 2 * Math.PI;
      P.push([rt * Math.cos(phi), h, rt * Math.sin(phi)]);
    }
    for (let i = 0; i < segments; i++) {           // seg..2seg-1: bottom ring
      const phi = (i / segments) * 2 * Math.PI;
      P.push([rb * Math.cos(phi), -h, rb * Math.sin(phi)]);
    }
    const ct = P.length; P.push([0,  h, 0]);       // top-cap center
    const cb = P.length; P.push([0, -h, 0]);       // bottom-cap center

    const I = [];
    for (let i = 0; i < segments; i++) {
      const j = (i + 1) % segments;
      const a = i, d = j;                          // top ring
      const b = segments + i, c = segments + j;    // bottom ring
      I.push(a, c, b,  a, d, c);                   // side wall (outward)
      I.push(ct, d, a);                            // top cap    (+y)
      I.push(cb, b, c);                            // bottom cap (-y)
    }
    // Zero-radius rings produce degenerate triangles, dropped in build().
    return this.addTriangles(P, I, opts);
  }

  // UV sphere centered at origin.
  // opts: { radius, segments (around), rings (top->bottom), translate, ... }
  addSphere(opts) {
    const R        = (opts && opts.radius)   || 0.5;
    const segments = (opts && opts.segments) || 32;
    const rings    = (opts && opts.rings)    || Math.max(8, segments >> 1);

    const P = [];
    for (let lat = 0; lat <= rings; lat++) {
      const theta = (lat / rings) * Math.PI;
      const y  = Math.cos(theta) * R;
      const sr = Math.sin(theta) * R;
      for (let seg = 0; seg <= segments; seg++) {
        const phi = (seg / segments) * 2 * Math.PI;
        P.push([sr * Math.cos(phi), y, sr * Math.sin(phi)]);
      }
    }
    const I = [];
    const stride = segments + 1;
    for (let lat = 0; lat < rings; lat++) {
      for (let seg = 0; seg < segments; seg++) {
        const a  = lat * stride + seg;       // this ring
        const b  = a + stride;               // next ring down
        const c  = b + 1;
        const dd = a + 1;
        // Degenerate pole triangles are dropped automatically in build().
        I.push(a, c, b);
        I.push(a, dd, c);
      }
    }
    return this.addTriangles(P, I, opts);
  }

  // --------------------------------------------------------------------------
  //  build() — weld, detect creases, apply overrides, smooth normals, emit
  //
  //  opts: {
  //    creaseAngleDeg: 40,        dihedral threshold for auto detection
  //    weldEps:        1e-4,      vertex welding tolerance
  //    overrides:      [],        [{ a:[xyz], b:[xyz], hard:bool }, ...]
  //    overrideEps:    1e-3,      endpoint matching tolerance for overrides
  //  }
  // --------------------------------------------------------------------------
  build(opts) {
    opts = opts || {};
    const creaseDeg = opts.creaseAngleDeg !== undefined ? opts.creaseAngleDeg : 40;
    const eps       = opts.weldEps     || 1e-4;
    const ovEps     = opts.overrideEps || 1e-3;
    const cosCrease = Math.cos(creaseDeg * Math.PI / 180);

    const ovMap = new Map();
    if (opts.overrides) {
      for (const o of opts.overrides)
        ovMap.set(ShapeBuilder.edgeKey(o.a, o.b, ovEps), !!o.hard);
    }

    // --- 1) Weld coincident positions so shared edges are detectable -------
    const positions = [];
    const posMap    = new Map();
    const weld = (x, y, z) => {
      const key = Math.round(x / eps) + "," +
                  Math.round(y / eps) + "," +
                  Math.round(z / eps);
      let idx = posMap.get(key);
      if (idx === undefined) {
        idx = positions.length;
        positions.push([x, y, z]);
        posMap.set(key, idx);
      }
      return idx;
    };

    // --- 2) Face list (degenerates dropped), face normals + areas ----------
    const faces      = [];
    const faceNormal = [];
    const faceArea   = [];
    const T = this.tris;
    for (let i = 0; i < T.length; i += 9) {
      const i0 = weld(T[i],   T[i+1], T[i+2]);
      const i1 = weld(T[i+3], T[i+4], T[i+5]);
      const i2 = weld(T[i+6], T[i+7], T[i+8]);
      if (i0 === i1 || i1 === i2 || i2 === i0) continue;
      const n = sb_cross(sb_sub(positions[i1], positions[i0]),
                         sb_sub(positions[i2], positions[i0]));
      const area = 0.5 * sb_len(n);
      if (area < 1e-12) continue;
      faces.push([i0, i1, i2]);
      faceNormal.push(sb_norm(n));
      faceArea.push(area);
    }

    // --- 3) Edge map: welded vertex pair -> adjacent faces ------------------
    const edges = new Map();
    for (let f = 0; f < faces.length; f++) {
      const tri = faces[f];
      for (let e = 0; e < 3; e++) {
        const a = tri[e], b = tri[(e + 1) % 3];
        const key = Math.min(a, b) + "," + Math.max(a, b);
        let rec = edges.get(key);
        if (!rec) { rec = { a, b, faces: [] }; edges.set(key, rec); }
        rec.faces.push(f);
      }
    }

    // --- 4) Classify edges (auto + overrides); union faces across SOFT ------
    const parent = new Array(faces.length);
    for (let i = 0; i < parent.length; i++) parent[i] = i;
    const find = (x) => {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };
    const union = (x, y) => { parent[find(x)] = find(y); };

    const hardEdges = [];   // { a, b, normal }  (welded indices)
    const softPairs = [];   // contour candidates: { a, b, nA, nB }
    const edgesOut  = [];   // exported edge list for tooling / the editor
    const COPLANAR  = Math.cos(1.0 * Math.PI / 180);   // skip ~flat edges
    for (const rec of edges.values()) {
      const pa = positions[rec.a], pb = positions[rec.b];

      let normalSum = [0, 0, 0];
      for (const f of rec.faces) normalSum = sb_add(normalSum, faceNormal[f]);
      let n = sb_norm(normalSum);
      if (sb_len(n) < 0.5) n = faceNormal[rec.faces[0]];

      // Auto classification
      let autoHard;
      if (rec.faces.length === 2) {
        autoHard = sb_dot(faceNormal[rec.faces[0]],
                          faceNormal[rec.faces[1]]) < cosCrease;
      } else {
        autoHard = true;   // boundary or non-manifold
      }

      // Override, if any
      const ov = ovMap.get(ShapeBuilder.edgeKey(pa, pb, ovEps));
      const hard = ov !== undefined ? ov : autoHard;

      if (!hard && rec.faces.length === 2) {
        union(rec.faces[0], rec.faces[1]);
        // A soft edge can become a silhouette when its faces straddle the
        // view plane — but only if they aren't (nearly) coplanar, like the
        // diagonals of triangulated flat quads, which never silhouette.
        const nA = faceNormal[rec.faces[0]], nB = faceNormal[rec.faces[1]];
        if (sb_dot(nA, nB) < COPLANAR)
          softPairs.push({ a: rec.a, b: rec.b, nA, nB });
      }
      if (hard)
        hardEdges.push({ a: rec.a, b: rec.b, normal: n });

      edgesOut.push({
        a: [pa[0], pa[1], pa[2]],
        b: [pb[0], pb[1], pb[2]],
        hard:       hard,
        auto:       autoHard,
        overridden: ov !== undefined,
        normal:     n,
      });
    }

    // --- 5) Smooth vertex normals within each (vertex, smoothing group) ----
    const groupNormals = new Map();
    for (let f = 0; f < faces.length; f++) {
      const g = find(f);
      const w = sb_scale(faceNormal[f], faceArea[f]);
      for (const v of faces[f]) {
        const key = v + "|" + g;
        const acc = groupNormals.get(key);
        groupNormals.set(key, acc ? sb_add(acc, w) : w);
      }
    }

    // --- 6) Emit surface vertices, deduped per (vertex, group) --------------
    const surfVerts   = [];
    const surfIndices = [];
    const outIndex    = new Map();
    for (let f = 0; f < faces.length; f++) {
      const g = find(f);
      for (const v of faces[f]) {
        const key = v + "|" + g;
        let idx = outIndex.get(key);
        if (idx === undefined) {
          idx = surfVerts.length / 6;
          const p = positions[v];
          const n = sb_norm(groupNormals.get(key));
          surfVerts.push(p[0], p[1], p[2], n[0], n[1], n[2]);
          outIndex.set(key, idx);
        }
        surfIndices.push(idx);
      }
    }

    // --- 7) Emit crease-line geometry for hard edges -------------------------
    // Each line vertex stores, in the v_normal slot, the OUTWARD end
    // direction of its segment (away from the other endpoint): -d at p0,
    // +d at p1. The line shader projects it to screen space and extrudes
    // perpendicular to it, so stroke thickness is constant regardless of
    // the edge's orientation to the camera; it also doubles as a square
    // end cap. Doubled length (dot > 1.5) still encodes the quad side.
    // Each hard edge becomes 4 vertices and 4 triangles (both windings,
    // so face culling never eats the line).
    const lineVerts   = [];
    const lineIndices = [];
    for (const e of hardEdges) {
      const base = lineVerts.length / 6;
      const p0 = positions[e.a], p1 = positions[e.b];
      const d  = sb_norm(sb_sub(p1, p0));
      lineVerts.push(p0[0], p0[1], p0[2], -2*d[0], -2*d[1], -2*d[2]);
      lineVerts.push(p0[0], p0[1], p0[2],   -d[0],   -d[1],   -d[2]);
      lineVerts.push(p1[0], p1[1], p1[2],    d[0],    d[1],    d[2]);
      lineVerts.push(p1[0], p1[1], p1[2],  2*d[0],  2*d[1],  2*d[2]);
      lineIndices.push(base+0, base+1, base+2,  base+2, base+1, base+3,
                       base+0, base+2, base+1,  base+2, base+3, base+1);
    }

    // --- 8) Contour (silhouette) candidates for view-dependent outlines ----
    // The outline of a smooth surface (e.g. the sides of a cylinder wall)
    // is not a fixed mesh edge: it moves as the camera turns. We therefore
    // emit every non-flat soft edge as a CANDIDATE carrying both adjacent
    // face normals; the contour shader tests them per frame and collapses
    // quads whose faces don't straddle the view plane.
    // Layout: 12 floats/vertex [pos(3), dir(3, format-2 encoding), nA(3), nB(3)]
    const contourVerts   = [];
    const contourIndices = [];
    if (opts.contours !== false) {
      for (const s of softPairs) {
        const base = contourVerts.length / 12;
        const p0 = positions[s.a], p1 = positions[s.b];
        const d  = sb_norm(sb_sub(p1, p0));
        const nA = s.nA, nB = s.nB;
        contourVerts.push(p0[0],p0[1],p0[2], -2*d[0],-2*d[1],-2*d[2], nA[0],nA[1],nA[2], nB[0],nB[1],nB[2]);
        contourVerts.push(p0[0],p0[1],p0[2],   -d[0],  -d[1],  -d[2], nA[0],nA[1],nA[2], nB[0],nB[1],nB[2]);
        contourVerts.push(p1[0],p1[1],p1[2],    d[0],   d[1],   d[2], nA[0],nA[1],nA[2], nB[0],nB[1],nB[2]);
        contourVerts.push(p1[0],p1[1],p1[2],  2*d[0], 2*d[1], 2*d[2], nA[0],nA[1],nA[2], nB[0],nB[1],nB[2]);
        contourIndices.push(base+0, base+1, base+2,  base+2, base+1, base+3,
                            base+0, base+2, base+1,  base+2, base+3, base+1);
      }
    }

    return {
      surfVerts:      new Float32Array(surfVerts),
      surfIndices:    new Uint32Array(surfIndices),
      lineVerts:      new Float32Array(lineVerts),
      lineIndices:    new Uint32Array(lineIndices),
      contourVerts:   new Float32Array(contourVerts),
      contourIndices: new Uint32Array(contourIndices),
      edges:          edgesOut,
      stats: {
        surfaceVertices: surfVerts.length / 6,
        triangles:       surfIndices.length / 3,
        edges:           edgesOut.length,
        hardEdges:       hardEdges.length,
        contourEdges:    contourVerts.length / 48,
        overrides:       ovMap.size,
      },
    };
  }
}
