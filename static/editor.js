const state = {};

function initEditor(canvas) {
    const id = canvas.id.split("-")[1];
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const maskImg = new Image();
        maskImg.crossOrigin = "anonymous";
        maskImg.onload = () => {
            ctx.globalAlpha = 0.45;
            ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
            state[id] = { drawing: false, brush: 20, ctx, canvas };
        };
        maskImg.src = canvas.dataset.mask;
    };
    img.src = canvas.dataset.image;

    canvas.addEventListener("mousedown", e => startDraw(id, e));
    canvas.addEventListener("mousemove", e => draw(id, e));
    window.addEventListener("mouseup", () => stopDraw(id));
}

function _pos(canvas, e) {
    const r = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - r.left) * canvas.width / r.width,
        y: (e.clientY - r.top) * canvas.height / r.height,
    };
}

function startDraw(id, e) {
    const s = state[id]; if (!s) return;
    s.drawing = true;
    draw(id, e);
}
function draw(id, e) {
    const s = state[id]; if (!s || !s.drawing) return;
    const p = _pos(s.canvas, e);
    s.ctx.fillStyle = "rgba(255, 0, 0, 0.6)";
    s.ctx.beginPath();
    s.ctx.arc(p.x, p.y, s.brush, 0, Math.PI * 2);
    s.ctx.fill();
}
function stopDraw(id) {
    const s = state[id]; if (s) s.drawing = false;
}
function setBrush(id, size) {
    if (state[id]) state[id].brush = size;
}
function clearCanvas(id) {
    const s = state[id]; if (!s) return;
    const img = new Image();
    img.onload = () => s.ctx.drawImage(img, 0, 0);
    img.src = s.canvas.dataset.image;
}
function saveMask(id) {
    const s = state[id]; if (!s) return;
    const dataURL = s.canvas.toDataURL("image/png");
    fetch(`/image/${id}/mask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask: dataURL })
    }).then(r => r.json()).then(res => {
        const el = document.getElementById(`status-${id}`);
        el.textContent = res.ok ? "✅ Маска сохранена" : "❌ Ошибка: " + (res.error || "");
        el.className = res.ok ? "small mt-2 text-success" : "small mt-2 text-danger";
    });
}

function sendToTraining(imgId) {
    const sel = document.getElementById(`verified-class-${imgId}`);
    const status = document.getElementById(`training-status-${imgId}`);
    if (!sel || !status) return;

    status.textContent = "Отправка...";
    status.className = "small mt-1 text-secondary";

    fetch(`/image/${imgId}/send-to-training`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verified_class: sel.value }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            status.textContent = "✅ " + (data.message || "Отправлено");
            status.className = "small mt-1 text-success";
        } else {
            status.textContent = "⚠ " + (data.error || "Ошибка");
            status.className = "small mt-1 text-warning";
        }
    })
    .catch(err => {
        status.textContent = "❌ " + err;
        status.className = "small mt-1 text-danger";
    });
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".mask-editor").forEach(initEditor);
});