"use strict";

fig3.addEventListener("pointerdown", (e) => {
  isDragging = true;
  lastMouseX = e.clientX;
  fig3.setPointerCapture(e.pointerId);
});

fig3.addEventListener("pointermove", (e) => {
  if (!isDragging) return;
  const dx = lastMouseX - e.clientX;
  lastMouseX = e.clientX;
  turntableAngle += dx * DRAG_SENSITIVITY;
});

fig3.addEventListener("pointerup", (e) => {
  isDragging = false;
  fig3.releasePointerCapture(e.pointerId);
});
