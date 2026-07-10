"use strict";

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

class ShapeBuilder {
  constructor() {
    this.tris = [];
  }

  static edgeKey(a, b, eps) {
    eps = eps || 1e-3;
    const q = (p) => Math.round(p[0]/eps) + "," +
                     Math.round(p[1]/eps) + "," +
                     Math.round(p[2]/eps);
    const ka = q(a), kb = q(b);
    return ka < kb ? ka + "|" + kb : kb + "|" + ka;
  }

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
        const a  = lat * stride + seg;
        const b  = a + stride;
        const c  = b + 1;
        const dd = a + 1;
        I.push(a, c, b);
        I.push(a, dd, c);
      }
    }
    return this.addTriangles(P, I, opts);
  }

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

    const parent = new Array(faces.length);
    for (let i = 0; i < parent.length; i++) parent[i] = i;
    const find = (x) => {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };
    const union = (x, y) => { parent[find(x)] = find(y); };

    const hardEdges = [];
    const edgesOut  = [];
    for (const rec of edges.values()) {
      const pa = positions[rec.a], pb = positions[rec.b];

      let normalSum = [0, 0, 0];
      for (const f of rec.faces) normalSum = sb_add(normalSum, faceNormal[f]);
      let n = sb_norm(normalSum);
      if (sb_len(n) < 0.5) n = faceNormal[rec.faces[0]];

      let autoHard;
      if (rec.faces.length === 2) {
        autoHard = sb_dot(faceNormal[rec.faces[0]],
                          faceNormal[rec.faces[1]]) < cosCrease;
      } else {
        autoHard = true;   // boundary or non-manifold
      }

      const ov = ovMap.get(ShapeBuilder.edgeKey(pa, pb, ovEps));
      const hard = ov !== undefined ? ov : autoHard;

      if (!hard && rec.faces.length === 2)
        union(rec.faces[0], rec.faces[1]);
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

    const lineVerts   = [];
    const lineIndices = [];
    for (const e of hardEdges) {
      const base = lineVerts.length / 6;
      const p0 = positions[e.a], p1 = positions[e.b];
      const n  = e.normal, n2 = sb_scale(e.normal, 2);
      lineVerts.push(p0[0], p0[1], p0[2],  n[0],  n[1],  n[2]);
      lineVerts.push(p0[0], p0[1], p0[2], n2[0], n2[1], n2[2]);
      lineVerts.push(p1[0], p1[1], p1[2],  n[0],  n[1],  n[2]);
      lineVerts.push(p1[0], p1[1], p1[2], n2[0], n2[1], n2[2]);
      lineIndices.push(base+0, base+1, base+2,  base+2, base+1, base+3,
                       base+0, base+2, base+1,  base+2, base+3, base+1);
    }

    return {
      surfVerts:   new Float32Array(surfVerts),
      surfIndices: new Uint32Array(surfIndices),
      lineVerts:   new Float32Array(lineVerts),
      lineIndices: new Uint32Array(lineIndices),
      edges:       edgesOut,
      stats: {
        surfaceVertices: surfVerts.length / 6,
        triangles:       surfIndices.length / 3,
        edges:           edgesOut.length,
        hardEdges:       hardEdges.length,
        overrides:       ovMap.size,
      },
    };
  }
}

