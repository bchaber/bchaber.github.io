"use strict";

const SURF_VERTS = [], SURF_INDICES = [], LINE_VERTS = [], LINE_INDICES = [];

// --- tiny vec3 helpers -------------------------------------------------------
const V = {
  sub: (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
  add: (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
  scl: (a, s) => [a[0]*s, a[1]*s, a[2]*s],
  dot: (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
  crs: (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
  nrm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1;
                return [a[0]/l, a[1]/l, a[2]/l]; },
};

// Displace the mid-node of edge k outward (⊥ to the edge, away from the
// element centroid at the origin) to make that single edge curved.
function bulgeEdge(pos, midIdx, [ca, cb], amount) {
  const d = V.nrm(V.sub(pos[cb], pos[ca]));
  const m = pos[midIdx];
  const out = V.nrm(V.sub(m, V.scl(d, V.dot(m, d))));
  pos[midIdx] = V.add(m, V.scl(out, amount));
}

// Scalar values at the element nodes: 1 at the curved edge's mid-node, 0 at
// the farthest node ("stress concentration" at the bulge). Interpolated with
// the quadratic basis functions like any FEM solution field — since the
// mid-node values are independent of the corners, the interpolation is
// genuinely quadratic (and may over/undershoot [0,1] between nodes).
function nodalField(pos, focus) {
  const d = pos.map(p => Math.hypot(p[0]-focus[0], p[1]-focus[1], p[2]-focus[2]));
  const lo = Math.min(...d), hi = Math.max(...d);
  const span = (hi - lo) || 1;
  return d.map(x => 1 - (x - lo) / span);
}

// Σ N_i(x) · nodal quantity, for vec3 positions and scalar field values.
const combine = (N, pts) => {
  const p = [0, 0, 0];
  N.forEach((n, i) => { p[0] += n*pts[i][0]; p[1] += n*pts[i][1]; p[2] += n*pts[i][2]; });
  return p;
};
const combine1 = (N, vals) => N.reduce((s, n, i) => s + n*vals[i], 0);

// --- Hex20 (serendipity brick), reference cube [-1,1]³ ------------------------
const HEX_CORNERS = [[-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
                     [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]];
const HEX_EDGES = [[0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4],
                   [0,4],[1,5],[2,6],[3,7]];

function hexElement(curvedEdge, bulge) {
  const refs = HEX_CORNERS.map(c => c.slice());
  for (const [a, b] of HEX_EDGES) refs.push(V.scl(V.add(refs[a], refs[b]), 0.5));
  const pos = refs.map(r => r.slice());
  bulgeEdge(pos, 8 + curvedEdge, HEX_EDGES[curvedEdge], bulge);
  const vals = nodalField(pos, pos[8 + curvedEdge]);
  const refreshField = () =>                       // field follows the nodes
    nodalField(pos, pos[8 + curvedEdge]).forEach((v, i) => vals[i] = v);

  const shape = (r, x) => {                      // serendipity shape functions
    const [a, b, c] = r;
    if (a && b && c)                             // corner node
      return 0.125 * (1+a*x[0]) * (1+b*x[1]) * (1+c*x[2])
                   * (a*x[0] + b*x[1] + c*x[2] - 2);
    return 0.25 * (a ? 1 + a*x[0] : 1 - x[0]*x[0])   // mid-edge node
                * (b ? 1 + b*x[1] : 1 - x[1]*x[1])
                * (c ? 1 + c*x[2] : 1 - x[2]*x[2]);
  };
  const shapes  = (x) => refs.map(r => shape(r, x));
  const evalXi  = (x) => combine(shapes(x), pos);
  const evalVal = (x) => combine1(shapes(x), vals);
  const evalPosVal = (x) => {                    // one basis evaluation for both
    const N = shapes(x), p = combine(N, pos);
    p.push(combine1(N, vals));
    return p;                                    // [x, y, z, field]
  };

  const faces = [];
  for (let ax = 0; ax < 3; ax++)
    for (const s of [-1, 1])
      faces.push({ tri: false, map: (u, v) => {
        const x = [0, 0, 0];
        x[ax] = s; x[(ax+1)%3] = -1 + 2*u; x[(ax+2)%3] = -1 + 2*v;
        return x;
      }});
  const edges = HEX_EDGES.map(([a, b], i) => [pos[a], pos[8+i], pos[b]]);
  return { pos, nc: 8, evalXi, evalVal, evalPosVal, refreshField, faces, edges };
}

// --- Tet10, reference tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1) ---------------------
const TET_CORNERS = [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]];  // physical
const TET_REF     = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]];
const TET_EDGES   = [[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]];
const TET_FACES   = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]];

function tetElement(curvedEdge, bulge) {
  const pos = TET_CORNERS.map(c => c.slice());
  for (const [a, b] of TET_EDGES) pos.push(V.scl(V.add(pos[a], pos[b]), 0.5));
  bulgeEdge(pos, 4 + curvedEdge, TET_EDGES[curvedEdge], bulge);
  const vals = nodalField(pos, pos[4 + curvedEdge]);
  const refreshField = () =>
    nodalField(pos, pos[4 + curvedEdge]).forEach((v, i) => vals[i] = v);

  const shapes = (x) => {                        // quadratic barycentric basis
    const L = [1 - x[0] - x[1] - x[2], x[0], x[1], x[2]];
    return L.map(l => l * (2*l - 1))
            .concat(TET_EDGES.map(([a, b]) => 4 * L[a] * L[b]));
  };
  const evalXi  = (x) => combine(shapes(x), pos);
  const evalVal = (x) => combine1(shapes(x), vals);
  const evalPosVal = (x) => {
    const N = shapes(x), p = combine(N, pos);
    p.push(combine1(N, vals));
    return p;
  };

  const faces = TET_FACES.map(([i, j, k]) => ({ tri: true, map: (u, v) => [
    TET_REF[i][0] + u*(TET_REF[j][0]-TET_REF[i][0]) + v*(TET_REF[k][0]-TET_REF[i][0]),
    TET_REF[i][1] + u*(TET_REF[j][1]-TET_REF[i][1]) + v*(TET_REF[k][1]-TET_REF[i][1]),
    TET_REF[i][2] + u*(TET_REF[j][2]-TET_REF[i][2]) + v*(TET_REF[k][2]-TET_REF[i][2]),
  ]}));
  const edges = TET_EDGES.map(([a, b], k) => [pos[a], pos[4+k], pos[b]]);
  return { pos, nc: 4, evalXi, evalVal, evalPosVal, refreshField, faces, edges };
}

// --- face tessellation --------------------------------------------------------
//  n×n grid (quad) or n-row barycentric grid (tri) over the face parameters.
//  Normals come from central differences of the quadratic mapping — smooth
//  across the face, so interior edges never show. Winding + normal are
//  flipped together so faces point away from the element centroid.
//  tr = { s: uniform scale, t: translation } (leaves normals unchanged).
function buildFace(el, face, n, tr, centroid) {
  const eps = 1e-3;
  const P = (u, v) => el.evalXi(face.map(u, v));
  // Forward differences reusing the vertex's own evaluation: same direction
  // as central differences to O(eps), at half the cost.
  const normalAt = (u, v, p0) => V.nrm(V.crs(V.sub(P(u+eps, v), p0),
                                             V.sub(P(u, v+eps), p0)));
  const [cu, cv] = face.tri ? [1/3, 1/3] : [0.5, 0.5];
  const pc = P(cu, cv);
  const flip = V.dot(normalAt(cu, cv, pc), V.sub(pc, centroid)) < 0 ? -1 : 1;

  const base = SURF_VERTS.length / 7;
  const rowStart = [];
  for (let i = 0, count = 0; i <= n; i++) {
    rowStart.push(count);
    for (let j = 0; j <= (face.tri ? n - i : n); j++, count++) {
      const q = el.evalPosVal(face.map(i/n, j/n));      // [x, y, z, field]
      const m = V.scl(normalAt(i/n, j/n, q), flip);
      SURF_VERTS.push(q[0]*tr.s + tr.t[0], q[1]*tr.s + tr.t[1], q[2]*tr.s + tr.t[2],
                      m[0], m[1], m[2], q[3]);
    }
  }
  const idx  = (i, j)    => base + rowStart[i] + j;
  const push = (a, b, c) => flip > 0 ? SURF_INDICES.push(a, b, c)
                                     : SURF_INDICES.push(a, c, b);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < (face.tri ? n - i : n); j++)
      if (face.tri) {
        push(idx(i, j), idx(i+1, j), idx(i, j+1));
        if (j < n - i - 1) push(idx(i+1, j), idx(i+1, j+1), idx(i, j+1));
      } else {
        push(idx(i, j), idx(i+1, j), idx(i+1, j+1));
        push(idx(i, j), idx(i+1, j+1), idx(i, j+1));
      }
}

// --- hard-edge lines -----------------------------------------------------------
//  Each edge is the 1D quadratic Lagrange curve through (corner, mid, corner),
//  sampled into m segments and packed as screen-space stroke quads: per
//  segment 4 verts whose "normal" holds the outward end direction (unit for
//  one quad side, doubled for the other — the encoding LINE_VS expects).
function addLineSeg(p0, p1) {
  const d = V.nrm(V.sub(p1, p0));
  const b = LINE_VERTS.length / 7;
  LINE_VERTS.push(p0[0], p0[1], p0[2],   -d[0],   -d[1],   -d[2], 0,
                  p0[0], p0[1], p0[2], -2*d[0], -2*d[1], -2*d[2], 0,
                  p1[0], p1[1], p1[2],    d[0],    d[1],    d[2], 0,
                  p1[0], p1[1], p1[2],  2*d[0],  2*d[1],  2*d[2], 0);
  LINE_INDICES.push(b, b+1, b+2,  b, b+2, b+3);
}

function buildEdge([pa, pm, pb], m, tr) {
  const pts = [];
  for (let k = 0; k <= m; k++) {
    const t = k/m, w0 = (1-t)*(1-2*t), w1 = 4*t*(1-t), w2 = t*(2*t-1);
    pts.push([(w0*pa[0] + w1*pm[0] + w2*pb[0]) * tr.s + tr.t[0],
              (w0*pa[1] + w1*pm[1] + w2*pb[1]) * tr.s + tr.t[1],
              (w0*pa[2] + w1*pm[2] + w2*pb[2]) * tr.s + tr.t[2]]);
  }
  for (let k = 0; k < m; k++) addLineSeg(pts[k], pts[k+1]);
}

function buildElement(el, tr, nFace, nEdge) {
  let c = [0, 0, 0];                         // live centroid from the corners
  for (let i = 0; i < el.nc; i++) c = V.add(c, el.pos[i]);
  c = V.scl(c, 1 / el.nc);
  for (const f of el.faces) buildFace(el, f, nFace, tr, c);
  for (const e of el.edges) buildEdge(e, nEdge, tr);
}

// --- scene: one Hex20 and one Tet10 side by side on the turntable -------------
//  Node positions are mutable; call rebuildScene() after editing them and let
//  the renderer re-upload (topology — index data and counts — never changes).
const SCENE = [
  { el: hexElement(1, 0.1),  tr: { s: 0.50, t: [-0.95, 0, 0] }, nFace: 12, nEdge: 10 },
  { el: tetElement(0, 0.1), tr: { s: 0.55, t: [ 0.95, 0, 0] }, nFace: 14, nEdge: 10 },
];

// Flattened node table for picking/editing; per element the corner nodes
// come first, then the mid-edge nodes.
const NODES = [];
// SCENE.forEach(({ el }, ei) =>
//   el.pos.forEach((_, ni) => NODES.push({ ei, ni, mid: ni >= el.nc })));

function nodeWorldPos(k) {
  const { ei, ni } = NODES[k], { el, tr } = SCENE[ei];
  return V.add(V.scl(el.pos[ni], tr.s), tr.t);
}

function setNodeWorld(k, w) {                // mutate in place: edge/eval refs stay live
  const { ei, ni } = NODES[k], { el, tr } = SCENE[ei];
  const p = el.pos[ni];
  p[0] = (w[0] - tr.t[0]) / tr.s;
  p[1] = (w[1] - tr.t[1]) / tr.s;
  p[2] = (w[2] - tr.t[2]) / tr.s;
}

function rebuildScene() {
  SURF_VERTS.length = SURF_INDICES.length = 0;
  LINE_VERTS.length = LINE_INDICES.length = 0;
  for (const { el, tr, nFace, nEdge } of SCENE) {
    el.refreshField();
    buildElement(el, tr, nFace, nEdge);
  }
}
rebuildScene();
