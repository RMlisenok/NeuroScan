const editors = {};

function getEditor(id) {
    if (editors[id]) return editors[id];

    const canvas = document.getElementById(`canvas-${id}`);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    const state = {
        canvas, ctx, img: null,
        drawing: false, brush: 10,
        lastX: 0, lastY: 0,
        loaded: false,
    };

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = canvas.dataset.image;
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        state.loaded = true;
    };
    state.img = img;

    canvas.addEventListener('mousedown', (e) => {
        if (!state.loaded) return;
        state.drawing = true;
        const p = toCanvasCoords(e, canvas);
        state.lastX = p.x;
        state.lastY = p.y;
        drawDot(state, p.x, p.y);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.drawing) return;
        const p = toCanvasCoords(e, canvas);
        drawLine(state, state.lastX, state.lastY, p.x, p.y);
        state.lastX = p.x;
        state.lastY = p.y;
    });

    const stop = () => { state.drawing = false; };
    canvas.addEventListener('mouseup', stop);
    canvas.addEventListener('mouseleave', stop);

    editors[id] = state;
    return state;
}

function toCanvasCoords(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function drawDot(state, x, y) {
    state.ctx.beginPath();
    state.ctx.arc(x, y, state.brush / 2, 0, Math.PI * 2);
    state.ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    state.ctx.fill();
}

function drawLine(state, x1, y1, x2, y2) {
    state.ctx.beginPath();
    state.ctx.lineCap = 'round';
    state.ctx.lineJoin = 'round';
    state.ctx.lineWidth = state.brush;
    state.ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    state.ctx.moveTo(x1, y1);
    state.ctx.lineTo(x2, y2);
    state.ctx.stroke();
}

function setBrush(id, size) {
    const s = getEditor(id);
    if (s) s.brush = size;
}

function clearCanvas(id) {
    const s = getEditor(id);
    if (!s || !s.img.complete) return;
    s.ctx.clearRect(0, 0, s.canvas.width, s.canvas.height);
    s.ctx.drawImage(s.img, 0, 0);
}

function saveMask(id) {
    const s = getEditor(id);
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
        getEditor(id);
    });
});