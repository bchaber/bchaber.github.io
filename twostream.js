/* ------------------------------------------------------------------
 * Two-Stream Instability — 1D Particle-In-Cell (PIC) simulation
 * ------------------------------------------------------------------
 * Renders a live phase-space plot (x vs v) on an HTML canvas.
 * Two cold electron beams drift in opposite directions through a
 * neutralizing ion background; the streams are unstable and the
 * classic phase-space "vortex" structure forms.
 *
 * Units are normalized:  ω_p = 1, q_e = -1, m_e = 1, ε_0 = 1.
 * The simulation is intentionally simplified (NGP deposition,
 * naive O(Ng^2) Poisson solve) — clarity over rigor.
 *
 * Usage:
 *   <canvas id="phase" width="900" height="500"></canvas>
 *   <script src="two_stream.js"></script>
 *   <script>TwoStream.start(document.getElementById('phase'));</script>
 * ------------------------------------------------------------------ */

(function (global) {
  'use strict';

  // ---------- Default simulation parameters ----------
  const DEFAULTS = {
    L:   50.0,    // domain length
    Ng:  64,      // number of grid cells
    Np:  4000,    // particles PER STREAM (total = 2*Np)
    v0:  1.0,     // beam drift speed
    vth: 0.05,    // small thermal spread (avoids a perfectly cold beam)
    dt:  0.05,    // time step
    // Display:
    vRange:  3.0,         // y-axis half-range for v
    colorA:  '#1a4fa3',   // stream A (drifting +v0)
    colorB:  '#b8341a',   // stream B (drifting -v0)
    bg:      '#fefefe',
    axis:    '#888',
    grid:    '#eee',
    text:    '#333',
  };

  // ---------- Helpers ----------
  function gaussian() {
    // Box–Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function wrap(x, L) {
    // periodic wrap into [0, L)
    x = x % L;
    if (x < 0) x += L;
    return x;
  }

  // ---------- Simulation state ----------
  function createSim(opts) {
    const p = Object.assign({}, DEFAULTS, opts || {});
    const N = 2 * p.Np;
    const dx = p.L / p.Ng;

    // Particle arrays (flat Float32 for speed)
    const x      = new Float32Array(N);
    const v      = new Float32Array(N);
    const stream = new Uint8Array(N); // 0 = A, 1 = B

    // Initialize: two counter-streaming beams, uniform in x,
    // with a tiny sinusoidal perturbation to seed the instability.
    const k_seed = 2 * Math.PI / p.L * 1; // fundamental mode
    const amp    = 0.01 * p.L;            // small position perturbation

    for (let i = 0; i < p.Np; i++) {
      // Stream A: +v0
      const xa = (i + 0.5) * (p.L / p.Np);
      x[i]      = wrap(xa + amp * Math.sin(k_seed * xa), p.L);
      v[i]      =  p.v0 + p.vth * gaussian();
      stream[i] = 0;
      // Stream B: -v0  (offset by half a particle spacing so they interleave)
      const j   = i + p.Np;
      const xb  = (i + 0.0) * (p.L / p.Np);
      x[j]      = wrap(xb - amp * Math.sin(k_seed * xb), p.L);
      v[j]      = -p.v0 + p.vth * gaussian();
      stream[j] = 1;
    }

    // Grid arrays
    const rho = new Float64Array(p.Ng); // charge density
    const phi = new Float64Array(p.Ng); // potential
    const E   = new Float64Array(p.Ng); // electric field at grid points

    // Per-particle charge so total electron charge cancels uniform ion bg.
    // Ions: density n0 = N/L, charge +1 per "ion". Electrons charge -q_e.
    // We pick q_e so that ε_0 = 1, ω_p = 1: ω_p^2 = n0 q^2/(ε_0 m) = 1
    // ⇒ q^2 = m/n0 = 1/(N/L) = L/N. So q_e = -sqrt(L/N).
    const qe = -Math.sqrt(p.L / N);

    return { p, N, dx, x, v, stream, rho, phi, E, qe, t: 0 };
  }

  // ---------- Charge deposition (nearest grid point) ----------
  function deposit(sim) {
    const { p, N, dx, x, rho, qe } = sim;
    const Ng = p.Ng;
    rho.fill(0);
    const invDx = 1 / dx;
    for (let i = 0; i < N; i++) {
      // CIC (cloud-in-cell) — slightly smoother than NGP, still cheap
      const xn = x[i] * invDx;
      const ig = Math.floor(xn);
      const f  = xn - ig;
      const i0 = ((ig % Ng) + Ng) % Ng;
      const i1 = (i0 + 1) % Ng;
      rho[i0] += qe * (1 - f);
      rho[i1] += qe * f;
    }
    // Normalize to density (charge per unit length) and add ion background +1
    const invDxArea = 1 / dx;
    for (let g = 0; g < Ng; g++) {
      rho[g] = rho[g] * invDxArea + 1.0;
    }
  }

  // ---------- Poisson solve via discrete Fourier transform ----------
  // Solves φ'' = -ρ on a periodic grid of size Ng. O(Ng^2) but Ng is small.
  function solvePoisson(sim) {
    const { p, dx, rho, phi } = sim;
    const Ng = p.Ng;

    // Forward DFT of rho → (Ar, Ai)
    const Ar = new Float64Array(Ng);
    const Ai = new Float64Array(Ng);
    for (let k = 0; k < Ng; k++) {
      let sr = 0, si = 0;
      const w = -2 * Math.PI * k / Ng;
      for (let n = 0; n < Ng; n++) {
        const a = w * n;
        sr += rho[n] * Math.cos(a);
        si += rho[n] * Math.sin(a);
      }
      Ar[k] = sr; Ai[k] = si;
    }

    // φ_k = ρ_k / k_eff^2,  k_eff^2 = (2/dx)^2 sin^2(π k / Ng)
    // (eigenvalue of the periodic 3-point Laplacian)
    const Pr = new Float64Array(Ng);
    const Pi = new Float64Array(Ng);
    Pr[0] = 0; Pi[0] = 0; // remove zero mode (charge-neutrality)
    for (let k = 1; k < Ng; k++) {
      const s = Math.sin(Math.PI * k / Ng);
      const k2 = (2 / dx) * (2 / dx) * s * s;
      Pr[k] = Ar[k] / k2;
      Pi[k] = Ai[k] / k2;
    }

    // Inverse DFT → phi
    for (let n = 0; n < Ng; n++) {
      let s = 0;
      const w = 2 * Math.PI * n / Ng;
      for (let k = 0; k < Ng; k++) {
        const a = w * k;
        s += Pr[k] * Math.cos(a) - Pi[k] * Math.sin(a);
      }
      phi[n] = s / Ng;
    }
  }

  // ---------- Field from potential (centered difference) ----------
  function computeField(sim) {
    const { p, dx, phi, E } = sim;
    const Ng = p.Ng;
    const inv2dx = 1 / (2 * dx);
    for (let g = 0; g < Ng; g++) {
      const gp = (g + 1) % Ng;
      const gm = (g - 1 + Ng) % Ng;
      E[g] = -(phi[gp] - phi[gm]) * inv2dx;
    }
  }

  // ---------- Push particles (leapfrog) ----------
  function push(sim) {
    const { p, N, dx, x, v, E, qe } = sim;
    const Ng = p.Ng;
    const dt = p.dt;
    const invDx = 1 / dx;

    for (let i = 0; i < N; i++) {
      // Interpolate E at particle position (CIC)
      const xn = x[i] * invDx;
      const ig = Math.floor(xn);
      const f  = xn - ig;
      const i0 = ((ig % Ng) + Ng) % Ng;
      const i1 = (i0 + 1) % Ng;
      const Ei = E[i0] * (1 - f) + E[i1] * f;

      // a = qe * E / m,  m = 1
      v[i] += qe * Ei * dt;
      x[i] = wrap(x[i] + v[i] * dt, p.L);
    }
  }

  function step(sim) {
    deposit(sim);
    solvePoisson(sim);
    computeField(sim);
    push(sim);
    sim.t += sim.p.dt;
  }

  // ---------- Rendering ----------
  function render(sim, ctx) {
    const { p, N, x, v, stream } = sim;
    const W = ctx.canvas._logicalW || ctx.canvas.width;
    const H = ctx.canvas._logicalH || ctx.canvas.height;

    // Plot area (margins for axes)
    const ML = 56, MR = 16, MT = 24, MB = 40;
    const PW = W - ML - MR;
    const PH = H - MT - MB;

    // Clear
    ctx.fillStyle = DEFAULTS.bg;
    ctx.fillRect(0, 0, W, H);

    // Plot background
    ctx.fillStyle = '#fbfaf7';
    ctx.fillRect(ML, MT, PW, PH);

    // Gridlines
    ctx.strokeStyle = DEFAULTS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < 10; i++) {
      const gx = ML + (PW * i) / 10;
      ctx.moveTo(gx, MT); ctx.lineTo(gx, MT + PH);
    }
    for (let j = 1; j < 6; j++) {
      const gy = MT + (PH * j) / 6;
      ctx.moveTo(ML, gy); ctx.lineTo(ML + PW, gy);
    }
    ctx.stroke();

    // Zero-velocity line
    const yZero = MT + PH / 2;
    ctx.strokeStyle = '#bbb';
    ctx.beginPath();
    ctx.moveTo(ML, yZero); ctx.lineTo(ML + PW, yZero);
    ctx.stroke();

    // --- Particle scatter on a REDUCED-resolution buffer ---
    // We rasterize each particle as a single pixel into a small offscreen
    // image, then drawImage() it scaled up to fill the plot area. This makes
    // particles visible (each "pixel" covers ~scale CSS px) and avoids
    // per-particle box loops.
    const RES_DIV = 2; // 1 small px == RES_DIV CSS px on screen
    const lowW = Math.max(1, (PW / RES_DIV) | 0);
    const lowH = Math.max(1, (PH / RES_DIV) | 0);

    // Reuse the offscreen canvas across frames
    if (!render._off || render._off.width !== lowW || render._off.height !== lowH) {
      const off = document.createElement('canvas');
      off.width  = lowW;
      off.height = lowH;
      render._off = off;
      render._offCtx = off.getContext('2d');
      render._offImg = render._offCtx.createImageData(lowW, lowH);
    }
    const offCtx = render._offCtx;
    const img    = render._offImg;
    const data   = img.data;
    // Clear to plot bg (#fbfaf7)
    for (let k = 0; k < data.length; k += 4) {
      data[k]   = 0xfb; data[k+1] = 0xfa; data[k+2] = 0xf7; data[k+3] = 0xff;
    }

    const cA = hexToRgb(p.colorA);
    const cB = hexToRgb(p.colorB);

    const sxScale = lowW / p.L;
    const syScale = lowH / (2 * p.vRange);
    const yMid = lowH / 2;

    for (let i = 0; i < N; i++) {
      const px = (x[i] * sxScale) | 0;
      let vy = v[i];
      if (vy >  p.vRange) vy =  p.vRange;
      if (vy < -p.vRange) vy = -p.vRange;
      const py = (yMid - vy * syScale) | 0;
      if (px < 0 || px >= lowW || py < 0 || py >= lowH) continue;
      const c = stream[i] === 0 ? cA : cB;
      const idx = (py * lowW + px) * 4;
      // alpha-ish blend so overlaps darken
      data[idx]     = (data[idx]     * 0.3 + c.r * 0.7) | 0;
      data[idx + 1] = (data[idx + 1] * 0.3 + c.g * 0.7) | 0;
      data[idx + 2] = (data[idx + 2] * 0.3 + c.b * 0.7) | 0;
      data[idx + 3] = 255;
    }
    offCtx.putImageData(img, 0, 0);

    // Scale the low-res buffer up onto the main canvas (smoothed off → crisp blocks)
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(render._off, 0, 0, lowW, lowH, ML, MT, PW, PH);
    ctx.imageSmoothingEnabled = true;

    // Axes frame
    ctx.strokeStyle = DEFAULTS.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ML + 0.5, MT + 0.5, PW, PH);

    // Tick labels
    ctx.fillStyle = DEFAULTS.text;
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
      const xv = (p.L * i) / 5;
      const px = ML + (PW * i) / 5;
      ctx.fillText(xv.toFixed(0), px, MT + PH + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    // Use CSS-pixel scale for labels (yMid/syScale above are in device px for ImageData)
    const yMidCss = MT + PH / 2;
    const syScaleCss = PH / (2 * p.vRange);
    for (let j = -2; j <= 2; j++) {
      const vv = (p.vRange * j) / 2;
      const py = yMidCss - vv * syScaleCss;
      ctx.fillText(vv.toFixed(1), ML - 6, py);
    }

    // Axis labels
    ctx.fillStyle = DEFAULTS.text;
    ctx.font = 'italic 13px "STIX Two Text", "Times New Roman", serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('x  (position)', ML + PW / 2, H - 10);
    ctx.save();
    ctx.translate(14, MT + PH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('v  (drift velocity)', 0, 0);
    ctx.restore();

    // Title + time + legend
    ctx.font = '12px "STIX Two Text", "Times New Roman", serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = DEFAULTS.text;
    ctx.fillText('Two-stream instability — phase space', ML, MT - 8);

    ctx.textAlign = 'right';
    ctx.fillText('t = ' + sim.t.toFixed(2) + ' / ωₚ⁻¹', ML + PW, MT - 8);

    // Legend
    const lx = ML + 12, ly = MT + 12;
    ctx.fillStyle = p.colorA;
    ctx.fillRect(lx, ly, 10, 10);
    ctx.fillStyle = p.colorB;
    ctx.fillRect(lx, ly + 16, 10, 10);
    ctx.fillStyle = DEFAULTS.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('+v₀', lx + 16, ly + 5);
    ctx.fillText('−v₀', lx + 16, ly + 21);
  }

  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }

  // ---------- Public API ----------
  function start(canvas, opts) {
    // Scale backing store for HiDPI (2× by default)
    const DPR = (opts && opts.dpr) || 2;
    const cssW = canvas.width;
    const cssH = canvas.height;
    canvas.style.width  = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width  = cssW * DPR;
    canvas.height = cssH * DPR;
    const ctx = canvas.getContext('2d');
    ctx.scale(DPR, DPR);
    Object.defineProperty(ctx.canvas, '_logicalW', { value: cssW });
    Object.defineProperty(ctx.canvas, '_logicalH', { value: cssH });

    const sim = createSim(opts);
    let running = (opts && opts.autoStart !== false);

    function frame() {
      if (!running) return;
      // a few sub-steps per frame for smoother dynamics
      const subSteps = 1;
      for (let s = 0; s < subSteps; s++) step(sim);
      render(sim, ctx);
      requestAnimationFrame(frame);
    }

    render(sim, ctx);
    requestAnimationFrame(frame);

    return {
      sim,
      isRunning: () => running,
      stop:   () => { running = false; },
      resume: () => { if (!running) { running = true; requestAnimationFrame(frame); } },
      reset:  (newOpts) => {
        const next = createSim(Object.assign({}, sim.p, newOpts || {}));
        Object.assign(sim, next);
        render(sim, ctx);
      },
    };
  }

  global.TwoStream = { start, createSim, step, render, DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);
