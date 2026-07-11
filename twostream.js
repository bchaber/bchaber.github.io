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
    aspect:  0.5,         // canvas height / width when sized by CSS
    maxDpr:  2,
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

  // ---------- Charge deposition (cloud-in-cell) ----------
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

    // Adaptive layout: shrink margins, ticks and labels on narrow canvases
    // (same tiers as tline.js so the two figures stay visually consistent).
    const compact = W < 420;
    const tiny    = W < 320;
    const ML = tiny ? 34 : compact ? 44 : 56;
    const MR = compact ? 10 : 16;
    const MT = compact ? 20 : 24;
    const MB = compact ? 32 : 40;
    const PW = W - ML - MR;
    const PH = H - MT - MB;
    const tickFont  = (compact ? 10 : 11) + 'px "JetBrains Mono", ui-monospace, monospace';
    const labelFont = (compact ? 'italic 11px' : 'italic 13px') + ' "STIX Two Text", "Times New Roman", serif';
    const titleFont = (compact ? '11px' : '12px') + ' "STIX Two Text", "Times New Roman", serif';
    const nXTicks   = tiny ? 2 : compact ? 3 : 5;

    // Clear
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, W, H);

    // Plot background
    ctx.fillStyle = '#fbfaf7';
    ctx.fillRect(ML, MT, PW, PH);

    // Gridlines
    ctx.strokeStyle = p.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const nVGrid = compact ? 6 : 10;
    for (let i = 1; i < nVGrid; i++) {
      const gx = ML + (PW * i) / nVGrid;
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
    // per-particle box loops. The buffer is recreated whenever the plot
    // area changes size, so it follows responsive re-layouts.
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
    // Clear to white
    for (let k = 0; k < data.length; k += 4) {
      data[k]   = 0xff; data[k+1] = 0xff; data[k+2] = 0xff; data[k+3] = 0xff;
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
    ctx.strokeStyle = p.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ML + 0.5, MT + 0.5, PW, PH);

    // Tick labels
    ctx.fillStyle = p.text;
    ctx.font = tickFont;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= nXTicks; i++) {
      const xv = (p.L * i) / nXTicks;
      const px = ML + (PW * i) / nXTicks;
      ctx.fillText(xv.toFixed(0), px, MT + PH + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yMidCss = MT + PH / 2;
    const syScaleCss = PH / (2 * p.vRange);
    const yStep = tiny ? 2 : 1;              // fewer y ticks when tiny
    for (let j = -2; j <= 2; j += yStep) {
      const vv = (p.vRange * j) / 2;
      const py = yMidCss - vv * syScaleCss;
      ctx.fillText(vv.toFixed(1), ML - 6, py);
    }

    // Axis labels
    ctx.fillStyle = p.text;
    ctx.font = labelFont;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(compact ? 'x' : 'x  (position)', ML + PW / 2, H - (compact ? 6 : 10));
    ctx.save();
    ctx.translate(compact ? 11 : 14, MT + PH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(compact ? 'v' : 'v  (drift velocity)', 0, 0);
    ctx.restore();

    // Title + time + legend
    ctx.font = titleFont;
    ctx.textAlign = 'left';
    ctx.fillStyle = p.text;
    ctx.fillText(compact ? 'Two-stream instability'
                         : 'Two-stream instability — phase space',
                 ML, MT - 8);

    ctx.textAlign = 'right';
    ctx.fillText('t = ' + sim.t.toFixed(2) + (compact ? '' : ' / ωₚ⁻¹'),
                 ML + PW, MT - 8);

    // Legend
    const lx = ML + 10, ly = MT + 10;
    ctx.fillStyle = p.colorA;
    ctx.fillRect(lx, ly, 10, 10);
    ctx.fillStyle = p.colorB;
    ctx.fillRect(lx, ly + 16, 10, 10);
    ctx.fillStyle = p.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = tickFont;
    ctx.fillText('+v₀', lx + 16, ly + 5);
    ctx.fillText('−v₀', lx + 16, ly + 21);
  }

  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }

  // Translucent "play" badge shown when the simulation is paused,
  // so the click-to-toggle affordance is visible on touch devices.
  function drawPlayBadge(ctx) {
    const W = ctx.canvas._logicalW || ctx.canvas.width;
    const H = ctx.canvas._logicalH || ctx.canvas.height;
    const cx = W / 2, cy = H / 2, r = Math.min(24, W * 0.07);
    ctx.save();
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = '#1d1b17';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#fffdf8';
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.28, cy - r * 0.42);
    ctx.lineTo(cx - r * 0.28, cy + r * 0.42);
    ctx.lineTo(cx + r * 0.46, cy);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // ---------- Public API ----------
  function start(canvas, opts) {
    opts = opts || {};
    const ctx = canvas.getContext('2d');
    const sim = createSim(opts);
    let running = opts.autoStart !== false;
    let rafId = 0;

    // Preferred aspect ratio: explicit option, else the width/height
    // attributes the canvas was authored with, else 1:2.
    const attrW = parseFloat(canvas.getAttribute('width'))  || 400;
    const attrH = parseFloat(canvas.getAttribute('height')) || 200;
    const aspect = opts.aspect || (attrH / attrW) || DEFAULTS.aspect;
    const maxDpr = opts.dpr || sim.p.maxDpr;

    // Size the drawing buffer from the CSS layout size (the stylesheet
    // sets width: 100%), times the device pixel ratio for sharpness.
    function layout() {
      const dpr = Math.min(global.devicePixelRatio || 1, maxDpr);
      const cssW = canvas.clientWidth ||
                   (canvas.parentElement && canvas.parentElement.clientWidth) ||
                   attrW;
      const cssH = Math.round(cssW * aspect);
      canvas.style.height = cssH + 'px';
      const pw = Math.max(1, Math.round(cssW * dpr));
      const ph = Math.max(1, Math.round(cssH * dpr));
      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      canvas._logicalW = cssW;
      canvas._logicalH = cssH;
    }

    function draw() {
      render(sim, ctx);
      if (!running) drawPlayBadge(ctx);
    }

    function frame() {
      if (!running) return;
      // a few sub-steps per frame for smoother dynamics
      const subSteps = opts.stepsPerFrame || 1;
      for (let s = 0; s < subSteps; s++) step(sim);
      render(sim, ctx);
      rafId = requestAnimationFrame(frame);
    }

    // Re-layout when the canvas' CSS size changes (rotation, window
    // resize, container reflow) and repaint at the new size.
    let ro = null;
    let onWinResize = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => { layout(); draw(); });
      ro.observe(canvas);
    } else if (typeof global.addEventListener === 'function') {
      onWinResize = () => { layout(); draw(); };
      global.addEventListener('resize', onWinResize);
    }

    layout();
    draw();
    if (running) rafId = requestAnimationFrame(frame);

    return {
      sim,
      isRunning: () => running,
      stop: () => {
        running = false;
        cancelAnimationFrame(rafId);
        draw();                 // repaint with the play badge
      },
      resume: () => {
        if (!running) {
          running = true;
          rafId = requestAnimationFrame(frame);
        }
      },
      reset: (newOpts) => {
        const next = createSim(Object.assign({}, sim.p, newOpts || {}));
        Object.assign(sim, next);
        draw();
      },
      destroy: () => {
        running = false;
        cancelAnimationFrame(rafId);
        if (ro) ro.disconnect();
        if (onWinResize) global.removeEventListener('resize', onWinResize);
      },
    };
  }

  global.TwoStream = { start, createSim, step, render, DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);
