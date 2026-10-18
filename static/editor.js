const editors = {};

function getCtx(id) {
    if (!editors[id]) {
        const canvas = document.getElementById(`canvas-${id}`);
        if (!canvas) return null;
        const img = new Image();
        img.src = canvas.dataset.image;
        const ctx = canvas.getContext('2d');
        const state = {
            canvas, ctx, img, drawing: false, brush: 10, eraser: false,
            lastX: 0, lastY: 0, loaded: false,
        };
        img.onload = () => {
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0);
            state.loaded = true;
        };
        canvas.addEventListener('mousedown', (e) => startDraw(e, id));
        canvas.addEventListener('mousemove', (e) => moveDraw(e, id));
        canvas.addEventListener('mouseup', () => stopDraw(id));
        canvas.addEventListener('mouseleave', () => stopDraw(id));
        editors[id] = state;
    }
    return editors[id];
}

function getPos(e, state) {
    const r = state.canvas.getBoundingClientRect();
    return {
        x: (e.clientX - r.left) * (state.canvas.width / r.width),
        y: (e.clientY - r.top) * (state.canvas.height / r.height),
    };
}

function startDraw(e, id) {
    const s = getCtx(id);
    if (!s || !s.loaded) return;
    s.drawing = true;
    const { x, y } = getPos(e, s);
    s.lastX = x; s.lastY = y;
    drawDot(s, x, y);
}

function moveDraw(e, id) {
    const s = getCtx(id);
    if (!s || !s.drawing) return;
    const { x, y } = getPos(e, s);
    drawLine(s, s.lastX, s.lastY, x, y);
    s.lastX = x; s.lastY = y;
}

function stopDraw(id) {
    const s = getCtx(id);
    if (s) s.drawing = false;
}

function drawDot(s, x, y) {
    s.ctx.beginPath();
    s.ctx.arc(x, y, s.brush / 2, 0, Math.PI * 2);
    s.ctx.fillStyle = s.eraser ? 'rgba(0,0,0,1)' : 'rgba(255,0,0,0.5)';
    s.ctx.globalCompositeOperation = s.eraser ? 'destination-out' : 'source-over';
    s.ctx.fill();
    s.ctx.globalCompositeOperation = 'source-over';
}

function drawLine(s, x1, y1, x2, y2) {
    s.ctx.beginPath();
    s.ctx.lineCap = 'round';
    s.ctx.lineJoin = 'round';
    s.ctx.lineWidth = s.brush;
    s.ctx.strokeStyle = s.eraser ? 'rgba(0,0,0,1)' : 'rgba(255,0,0,0.5)';
    s.ctx.globalCompositeOperation = s.eraser ? 'destination-out' : 'source-over';
    s.ctx.moveTo(x1, y1);
    s.ctx.lineTo(x2, y2);
    s.ctx.stroke();
    s.ctx.globalCompositeOperation = 'source-over';
}

function setBrush(id, size) {
    const s = getCtx(id);
    if (s) { s.brush = size; s.eraser = false; }
}

function setEraser(id) {
    const s = getCtx(id);
    if (s) { s.eraser = true; }
}

function clearCanvas(id) {
    const s = getCtx(id);
    if (!s) return;
    s.ctx.clearRect(0, 0, s.canvas.width, s.canvas.height);
    if (s.img && s.img.complete) s.ctx.drawImage(s.img, 0, 0);
}

function saveMask(id) {
    const s = getCtx(id);
    if (!s) return;
    const dataUrl = s.canvas.toDataURL('image/png');
    fetch(`/image/${id}/mask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mask: dataUrl }),
    })
    .then(r => r.json())
    .then(data => {
        const el = document.getElementById(`status-${id}`);
        if (data.ok) {
            el.innerHTML = '<span class="badge bg-success">Маска сохранена</span>';
        } else {
            el.innerHTML = `<span class="badge bg-danger">${data.error || 'Ошибка'}</span>`;
        }
    })
    .catch(() => {
        document.getElementById(`status-${id}`).innerHTML =
            '<span class="badge bg-danger">Ошибка сети</span>';
    });
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('canvas.mask-editor').forEach(c => {
        const id = c.id.replace('canvas-', '');
        getCtx(id);
    });
});