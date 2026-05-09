/* ------------------------------------------------------------------
 * 1D FDTD solver for a lossless transmission line
 * ------------------------------------------------------------------
 * Solves the telegrapher's equations on a Yee-style staggered grid:
 *
 *     ∂V/∂t = -(1/C) ∂I/∂x
 *     ∂I/∂t = -(1/L) ∂V/∂x
 *
 *   V (voltage) lives on integer nodes:        V[0]   V[1]   V[2] ...
 *   I (current) lives on half-integer nodes:       I[0]   I[1] ...
 *
 * Wave speed  c = 1/sqrt(L*C),
 * Char. imp.  Z0 = sqrt(L/C),
 * Courant     dt = S * dx/c,  S < 1 for stability.
 *
 * Source: Gaussian voltage pulse injected at the left boundary.
 * Right boundary: matched (1st-order absorbing) by default, or a
 * configurable resistive load to show reflections.
 *
 * Rendering: HTML canvas with a hover-revealed YouTube-style control
 * bar (handled by the host HTML).
 *
 * Usage:
 *   const handle = TLine.start(canvas, { autoStart: false });
 *   handle.resume(); handle.stop(); handle.reset();
 * ------------------------------------------------------------------ */

(function (global) {
  'use strict';

  const DEFAULTS = {
    // Physics
    Nx:    400,      // number of voltage nodes
    dx:    1.0,      // spatial step (arbitrary units)
    L_pul: 1.0,      // inductance per unit length
    C_pul: 1.0,      // capacitance per unit length
    S:     0.99,     // Courant number (<1 for stability)
    // Source
    pulseT0:    100,  // center of Gaussian pulse (in steps)
    pulseWidth: 25,   // sigma (in steps)
    pulseAmp:   0.5,
    // Termination at right end: 'matched' or a numeric R (load resistance)
    termination: 'matched',
    // Display
    vRange:  1.2,            // y-axis half-range for V
    colorV:  '#1a4fa3',
    colorI:  '#b8341a',
    bg:      '#fefefe',
    plotBg:  '#fbfaf7',
    axis:    '#888',
    grid:    '#eee',
    text:    '#333',
    showI:   true,           // overlay current trace
  };

  function createSim(opts) {
    const p = Object.assign({}, DEFAULTS, opts || {});
    const c = 1 / Math.sqrt(p.L_pul * p.C_pul);
    const Z0 = Math.sqrt(p.L_pul / p.C_pul);
    const dt = p.S * p.dx / c;

    // Voltage on Nx integer nodes, current on Nx-1 half-integer nodes.
    const V = new Float64Array(p.Nx);
    const I = new Float64Array(p.Nx - 1);

    // Coefficients
    const cV = dt / (p.C_pul * p.dx);
    const cI = dt / (p.L_pul * p.dx);

    return { p, V, I, c, Z0, dt, cV, cI, n: 0, t: 0 };
  }

  function step(sim) {
    const { p, V, I, cV, cI, Z0 } = sim;
    const Nx = p.Nx;

    // --- Update I (interior, between V[i] and V[i+1]) ---
    for (let i = 0; i < Nx - 1; i++) {
      I[i] -= cI * (V[i + 1] - V[i]);
    }

    // --- Update V (interior) ---
    for (let i = 1; i < Nx - 1; i++) {
      V[i] -= cV * (I[i] - I[i - 1]);
    }

    // --- Source: Gaussian voltage pulse at left end (hard source) ---
    const t0 = p.pulseT0, sg = p.pulseWidth;
    if (sim.n < 6 * sg + t0) {
      const arg = (sim.n - t0) / sg;
      V[0] = p.pulseAmp * Math.exp(-arg * arg);
    } else {
      // After pulse: matched absorbing boundary at the left to swallow
      // any reflection that comes back.
      V[0] = V[1] - Z0 * I[0];
    }

    // --- Right-end termination ---
    if (p.termination === 'matched') {
      // Matched load: V_N = Z0 * I_{N-1}  (1st-order absorbing)
      V[Nx - 1] = V[Nx - 2] + Z0 * I[Nx - 2];
      // Equivalent (and slightly more standard) ABC:
      //   V[Nx-1] = V[Nx-2] - cV*(0 - I[Nx-2]) but tuned to no-reflection.
    } else if (typeof p.termination === 'number') {
      // Resistive load R: V_end = R * I_end  →  reflection coeff (R-Z0)/(R+Z0)
      const R = p.termination;
      // Treat the last half-cell as terminated by R: enforce V_end = R*I_{N-1}
      // using a simple update consistent with the staggered scheme.
      V[Nx - 1] = (V[Nx - 2] + (R / Z0) * (V[Nx - 2] + Z0 * I[Nx - 2])) /
                  (1 + R / Z0);
      // Fallback simple form (works fine for visualization):
      // V[Nx-1] = R * I[Nx-2];
    } else if (p.termination === 'open') {
      // Open circuit: I at end = 0 → reflect with same sign
      V[Nx - 1] = V[Nx - 2] + cV * I[Nx - 2];
    } else if (p.termination === 'short') {
      V[Nx - 1] = 0;
    }

    sim.n++;
    sim.t += sim.dt;
  }

  // ---------- Rendering ----------
  function render(sim, ctx) {
    const { p, V, I, Z0 } = sim;
    const W = ctx.canvas._logicalW || ctx.canvas.width;
    const H = ctx.canvas._logicalH || ctx.canvas.height;

    const ML = 56, MR = 16, MT = 24, MB = 40;
    const PW = W - ML - MR;
    const PH = H - MT - MB;

    // Clear
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = p.plotBg;
    ctx.fillRect(ML, MT, PW, PH);

    // Gridlines
    ctx.strokeStyle = p.grid;
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

    // Zero line
    const yMid = MT + PH / 2;
    ctx.strokeStyle = '#bbb';
    ctx.beginPath();
    ctx.moveTo(ML, yMid); ctx.lineTo(ML + PW, yMid);
    ctx.stroke();

    // --- Plot V(x) ---
    const Nx = p.Nx;
    const sxScale = PW / (Nx - 1);
    const syScale = PH / (2 * p.vRange);

    function clamp(y) {
      if (y >  p.vRange) return  p.vRange;
      if (y < -p.vRange) return -p.vRange;
      return y;
    }

    ctx.strokeStyle = p.colorV;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < Nx; i++) {
      const px = ML + i * sxScale;
      const py = yMid - clamp(V[i]) * syScale;
      if (i === 0) ctx.moveTo(px, py);
      else         ctx.lineTo(px, py);
    }
    ctx.stroke();

    // --- Plot I(x) * Z0 (so current is on same scale as voltage) ---
    if (p.showI) {
      ctx.strokeStyle = p.colorI;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let i = 0; i < Nx - 1; i++) {
        const px = ML + (i + 0.5) * sxScale;
        const py = yMid - clamp(I[i] * Z0) * syScale;
        if (i === 0) ctx.moveTo(px, py);
        else         ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Frame
    ctx.strokeStyle = p.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ML + 0.5, MT + 0.5, PW, PH);

    // Tick labels
    ctx.fillStyle = p.text;
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
      const xv = ((Nx - 1) * p.dx * i) / 5;
      const px = ML + (PW * i) / 5;
      ctx.fillText(xv.toFixed(0), px, MT + PH + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let j = -2; j <= 2; j++) {
      const vv = (p.vRange * j) / 2;
      const py = yMid - vv * syScale;
      ctx.fillText(vv.toFixed(1), ML - 6, py);
    }

    // Axis labels
    ctx.fillStyle = p.text;
    ctx.font = 'italic 13px "STIX Two Text", "Times New Roman", serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('x  (position along line)', ML + PW / 2, H - 10);
    ctx.save();
    ctx.translate(14, MT + PH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('V (x, t),   Z₀ · I (x, t)', 0, 0);
    ctx.restore();

    // Title + time
    ctx.font = '12px "STIX Two Text", "Times New Roman", serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = p.text;
    ctx.fillText('1D FDTD — travelling pulse on a transmission line', ML, MT - 8);
    ctx.textAlign = 'right';
    ctx.fillText('t = ' + sim.t.toFixed(2), ML + PW, MT - 8);

    // Legend
    const lx = ML + 12, ly = MT + 12;
    ctx.fillStyle = p.colorV;
    ctx.fillRect(lx, ly, 14, 2);
    ctx.fillStyle = p.colorI;
    // dashed legend line for I
    for (let k = 0; k < 3; k++) {
      ctx.fillRect(lx + k * 5, ly + 16, 3, 2);
    }
    ctx.fillStyle = p.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText('V (x)',         lx + 20, ly + 1);
    ctx.fillText('Z₀ · I (x)',    lx + 20, ly + 17);
  }

  // ---------- Public API ----------
  function start(canvas, opts) {
    const DPR = (opts && opts.dpr) || 2;
    const cssW = canvas.width;
    const cssH = canvas.height;
    canvas.style.width  = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width  = cssW * DPR;
    canvas.height = cssH * DPR;
    const ctx = canvas.getContext('2d');
    ctx.scale(DPR, DPR);
    Object.defineProperty(ctx.canvas, '_logicalW', { value: cssW, configurable: true });
    Object.defineProperty(ctx.canvas, '_logicalH', { value: cssH, configurable: true });

    const sim = createSim(opts);
    let running = (opts && opts.autoStart !== false);

    function frame() {
      if (!running) return;
      // Multiple sub-steps per frame so the pulse moves visibly.
      const subSteps = (opts && opts.stepsPerFrame) || 2;
      for (let s = 0; s < subSteps; s++) step(sim);
      render(sim, ctx);
      requestAnimationFrame(frame);
    }
    render(sim, ctx);
    if (running) requestAnimationFrame(frame);

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

  global.TLine = { start, createSim, step, render, DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);