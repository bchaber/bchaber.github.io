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
 * Rendering: HTML canvas. The canvas is sized by CSS (width: 100%);
 * the drawing buffer follows the CSS size and the device pixel ratio,
 * and re-lays itself out via ResizeObserver, so the figure scales
 * correctly on narrow/mobile screens. Layout margins, tick counts and
 * labels adapt to the available width.
 *
 * Usage:
 *   const handle = TLine.start(canvas, { autoStart: false });
 *   handle.resume(); handle.stop(); handle.reset(); handle.destroy();
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
    // Termination at right end: 'matched', 'open', 'short', or numeric R
    termination: 'matched',
    // Display
    vRange:  1.2,            // y-axis half-range for V
    colorV:  '#1a4fa3',
    colorI:  '#b8341a',
    bg:      '#ffffff',
    plotBg:  '#ffffff',
    axis:    '#888',
    grid:    '#eee',
    text:    '#333',
    showI:   true,           // overlay current trace
    aspect:  0.5,            // canvas height / width when sized by CSS
    maxDpr:  2,
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
    } else if (typeof p.termination === 'number') {
      // Resistive load R: V_end = R * I_end  →  reflection coeff (R-Z0)/(R+Z0)
      const R = p.termination;
      V[Nx - 1] = (V[Nx - 2] + (R / Z0) * (V[Nx - 2] + Z0 * I[Nx - 2])) /
                  (1 + R / Z0);
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

    // Adaptive layout: shrink margins, ticks and labels on narrow canvases.
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
    ctx.fillStyle = p.plotBg;
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
    ctx.font = tickFont;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= nXTicks; i++) {
      const xv = ((Nx - 1) * p.dx * i) / nXTicks;
      const px = ML + (PW * i) / nXTicks;
      ctx.fillText(xv.toFixed(0), px, MT + PH + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yStep = tiny ? 2 : 1;              // fewer y ticks when tiny
    for (let j = -2; j <= 2; j += yStep) {
      const vv = (p.vRange * j) / 2;
      const py = yMid - vv * syScale;
      ctx.fillText(vv.toFixed(1), ML - 6, py);
    }

    // Axis labels
    ctx.fillStyle = p.text;
    ctx.font = labelFont;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(compact ? 'x' : 'x  (position along line)', ML + PW / 2, H - (compact ? 6 : 10));
    ctx.save();
    ctx.translate(compact ? 11 : 14, MT + PH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(compact ? 'V,  Z₀·I' : 'V (x, t),   Z₀ · I (x, t)', 0, 0);
    ctx.restore();

    // Title + time
    ctx.font = titleFont;
    ctx.textAlign = 'left';
    ctx.fillStyle = p.text;
    ctx.fillText(compact ? '1D FDTD — travelling pulse'
                         : '1D FDTD — travelling pulse on a transmission line',
                 ML, MT - 8);
    ctx.textAlign = 'right';
    ctx.fillText('t = ' + sim.t.toFixed(2), ML + PW, MT - 8);

    // Legend
    const lx = ML + 10, ly = MT + 10;
    ctx.fillStyle = p.colorV;
    ctx.fillRect(lx, ly, 14, 2);
    ctx.fillStyle = p.colorI;
    for (let k = 0; k < 3; k++) {           // dashed legend line for I
      ctx.fillRect(lx + k * 5, ly + 15, 3, 2);
    }
    ctx.fillStyle = p.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = tickFont;
    ctx.fillText('V (x)',       lx + 20, ly + 1);
    ctx.fillText('Z₀ · I (x)',  lx + 20, ly + 16);
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
      // Multiple sub-steps per frame so the pulse moves visibly.
      const subSteps = opts.stepsPerFrame || 2;
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

  global.TLine = { start, createSim, step, render, DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);
