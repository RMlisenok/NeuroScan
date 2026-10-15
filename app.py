import os
import io
import json
import uuid
import datetime
import numpy as np
import cv2
import torch
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, send_file, jsonify, abort, session)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from models_db import db, User, Patient, MRISession, MRIImage, AuditLog
from qc import check_image
from ml import (build_model, GradCAM, prepare_input, overlay_cam,
                largest_bbox, bbox_area_pct, CLASS_NAMES_RU)

# ---------- Инициализация ----------
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("instance", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------- Модель (ленивая загрузка) ----------
model = None
gradcam = None
CLASSES = []


def ensure_model():
    global model, gradcam, CLASSES
    if model is not None:
        return True
    path = app.config["MODEL_PATH"]
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, map_location=DEVICE)
    CLASSES = ckpt["classes"]
    model = build_model(num_classes=len(CLASSES), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    gradcam = GradCAM(model)
    return True


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))


def log_action(action, entity=None, entity_id=None, details=None):
    try:
        db.session.add(AuditLog(
            user_id=current_user.id if current_user.is_authenticated else None,
            action=action, entity=entity, entity_id=entity_id, details=details
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()


def role_required(role):
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return deco


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXT"]


# ---------- CLI: создать админа ----------
@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("БД создана.")


@app.cli.command("create-user")
def create_user_cli():
    import getpass
    username = input("username: ").strip()
    full_name = input("ФИО: ").strip()
    role = input("role (doctor/admin) [doctor]: ").strip() or "doctor"
    pw = getpass.getpass("password: ")
    if User.query.filter_by(username=username).first():
        print("Такой пользователь уже есть.")
        return
    u = User(username=username, full_name=full_name, role=role,
             password_hash=generate_password_hash(pw))
    db.session.add(u)
    db.session.commit()
    print(f"Пользователь {username} создан.")


# ---------- Auth ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            log_action("login", "user", user.id)
            return redirect(url_for("dashboard"))
        flash("Неверный логин или ключ доступа")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    log_action("logout", "user", current_user.id)
    logout_user()
    return redirect(url_for("login"))


# ---------- Dashboard ----------
@app.route("/")
@login_required
def dashboard():
    q = request.args.get("q", "").strip()
    query = Patient.query.filter_by(doctor_id=current_user.id)
    if q:
        query = query.filter(Patient.full_name.ilike(f"%{q}%"))
    patients = query.order_by(Patient.updated_at.desc()).all()
    return render_template("dashboard.html", patients=patients, q=q)


# ---------- Пациенты ----------
@app.route("/patient/new", methods=["GET", "POST"])
@login_required
def patient_new():
    if request.method == "POST":
        p = Patient(
            doctor_id=current_user.id,
            full_name=request.form["full_name"].strip(),
            birth_date=_parse_date(request.form.get("birth_date")),
            gender=request.form.get("gender"),
            phone=request.form.get("phone"),
            email=request.form.get("email"),
            address=request.form.get("address"),
            insurance_number=request.form.get("insurance_number"),
            anamnesis=request.form.get("anamnesis"),
            complaints=request.form.get("complaints"),
            notes=request.form.get("notes"),
        )
        db.session.add(p)
        db.session.commit()
        log_action("patient_create", "patient", p.id)
        flash("Пациент добавлен")
        return redirect(url_for("patient_card", pid=p.id))
    return render_template("patient_new.html")


@app.route("/patient/<int:pid>")
@login_required
def patient_card(pid):
    p = _get_patient_or_404(pid)
    sessions = MRISession.query.filter_by(patient_id=p.id) \
                              .order_by(MRISession.created_at.desc()).all()
    return render_template("patient_card.html", patient=p, sessions=sessions)


@app.route("/patient/<int:pid>/edit", methods=["POST"])
@login_required
def patient_edit(pid):
    p = _get_patient_or_404(pid)
    p.full_name = request.form.get("full_name", p.full_name)
    p.birth_date = _parse_date(request.form.get("birth_date"))
    p.gender = request.form.get("gender")
    p.phone = request.form.get("phone")
    p.email = request.form.get("email")
    p.address = request.form.get("address")
    p.insurance_number = request.form.get("insurance_number")
    p.anamnesis = request.form.get("anamnesis")
    p.complaints = request.form.get("complaints")
    p.notes = request.form.get("notes")
    db.session.commit()
    log_action("patient_update", "patient", p.id)
    flash("Данные обновлены")
    return redirect(url_for("patient_card", pid=p.id))


# ---------- МРТ сессии ----------
@app.route("/patient/<int:pid>/mri/new", methods=["GET"])
@login_required
def mri_upload_page(pid):
    p = _get_patient_or_404(pid)
    return render_template("mri_upload.html", patient=p,
                           max_images=app.config["MAX_IMAGES_PER_SESSION"])


@app.route("/patient/<int:pid>/mri/new", methods=["POST"])
@login_required
def mri_upload(pid):
    p = _get_patient_or_404(pid)
    files = request.files.getlist("files")
    files = [f for f in files if f and f.filename]

    if not files:
        flash("Не выбрано ни одного файла")
        return redirect(url_for("mri_upload_page", pid=pid))
    if len(files) > app.config["MAX_IMAGES_PER_SESSION"]:
        flash(f"Максимум {app.config['MAX_IMAGES_PER_SESSION']} снимков за сессию")
        return redirect(url_for("mri_upload_page", pid=pid))

    if not ensure_model():
        flash("Модель не загружена. Запустите train.py")
        return redirect(url_for("mri_upload_page", pid=pid))

    s = MRISession(patient_id=p.id)
    db.session.add(s)
    db.session.commit()

    folder = os.path.join(app.config["UPLOAD_FOLDER"], f"p{p.id}", f"s{s.id}")
    os.makedirs(folder, exist_ok=True)

    for f in files:
        if not allowed_file(f.filename):
            continue
        img_bytes = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        uid = uuid.uuid4().hex[:10]
        orig_path = os.path.join(folder, f"{uid}_orig.png")
        cv2.imwrite(orig_path, img_bgr)

        qc = check_image(img_bgr)

        rec = MRIImage(
            session_id=s.id,
            filename=secure_filename(f.filename),
            original_path=_rel(orig_path),
            qc_ok=qc.ok,
            qc_reasons=json.dumps(qc.reasons, ensure_ascii=False),
            qc_metrics=json.dumps(qc.metrics, ensure_ascii=False),
        )

        if qc.ok:
            _run_inference(img_bgr, folder, uid, rec)

        db.session.add(rec)

    db.session.commit()
    log_action("mri_upload", "session", s.id, f"{len(files)} files")
    return redirect(url_for("mri_result", sid=s.id))


@app.route("/session/<int:sid>")
@login_required
def mri_result(sid):
    s = _get_session_or_404(sid)
    images = MRIImage.query.filter_by(session_id=s.id).all()
    return render_template("mri_result.html", session=s, patient=s.patient, images=images)


@app.route("/image/<int:iid>/mask", methods=["POST"])
@login_required
def save_mask(iid):
    """Сохраняет отредактированную маску (PNG base64) и отправляет на дообучение."""
    img = _get_image_or_404(iid)
    data = request.get_json() or {}
    mask_b64 = data.get("mask")
    if not mask_b64:
        return jsonify({"ok": False, "error": "no mask"}), 400

    import base64
    raw = base64.b64decode(mask_b64.split(",")[-1])
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return jsonify({"ok": False, "error": "bad mask"}), 400

    folder = os.path.dirname(_abs(img.original_path))
    mask_path = os.path.join(folder, f"mask_{img.id}.png")
    cv2.imwrite(mask_path, mask)
    img.edited_mask_path = _rel(mask_path)
    img.session.status = "edited"
    db.session.commit()

    # TODO: положить в очередь на дообучение (active learning)
    log_action("mask_edit", "image", img.id)
    return jsonify({"ok": True})


@app.route("/session/<int:sid>/report")
@login_required
def session_report(sid):
    s = _get_session_or_404(sid)
    return _build_pdf(s)


@app.route("/patient/<int:pid>/history")
@login_required
def patient_history(pid):
    p = _get_patient_or_404(pid)
    sessions = MRISession.query.filter_by(patient_id=p.id) \
                              .order_by(MRISession.created_at.desc()).all()
    return render_template("history.html", patient=p, sessions=sessions)


# ---------- helpers ----------
def _parse_date(v):
    if not v:
        return None
    try:
        return datetime.datetime.strptime(v, "%Y-%m-%d").date()
    except ValueError:
        return None


def _rel(path):
    return os.path.relpath(path, os.path.abspath(os.path.dirname(__file__)))


def _abs(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)


def _get_patient_or_404(pid):
    p = Patient.query.get_or_404(pid)
    if p.doctor_id != current_user.id and current_user.role != "admin":
        abort(403)
    return p


def _get_session_or_404(sid):
    s = MRISession.query.get_or_404(sid)
    if s.patient.doctor_id != current_user.id and current_user.role != "admin":
        abort(403)
    return s


def _get_image_or_404(iid):
    img = MRIImage.query.get_or_404(iid)
    if img.session.patient.doctor_id != current_user.id and current_user.role != "admin":
        abort(403)
    return img


def _run_inference(img_bgr, folder, uid, rec):
    tensor, img_rgb_224 = prepare_input(img_bgr)
    tensor = tensor.to(DEVICE)
    cam, class_idx, probs = gradcam.generate(tensor)
    pred_class = CLASSES[class_idx]
    confidence = float(probs[class_idx])

    overlay, mask, contours = overlay_cam(img_rgb_224, cam, threshold=0.5)
    bbox = largest_bbox(contours, overlay.shape)
    area = bbox_area_pct(bbox, overlay.shape)

    overlay_path = os.path.join(folder, f"{uid}_overlay.png")
    mask_path = os.path.join(folder, f"{uid}_mask.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask)

    rec.overlay_path = _rel(overlay_path)
    rec.edited_mask_path = _rel(mask_path)
    rec.predicted_class = pred_class
    rec.confidence = confidence
    rec.probabilities = json.dumps(
        {CLASS_NAMES_RU.get(CLASSES[i], CLASSES[i]): float(probs[i])
         for i in range(len(CLASSES))}, ensure_ascii=False)
    rec.bbox = ",".join(map(str, bbox)) if bbox else None
    rec.area_pct = area
    rec.cam_threshold = 0.5


def _build_pdf(s):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, H - 50, "NeuroScan AI — отчёт по МРТ-исследованию")
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 70, f"Пациент: {s.patient.full_name}")
    c.drawString(40, H - 85, f"Дата рождения: {s.patient.birth_date or '-'}")
    c.drawString(40, H - 100, f"Сессия №{s.id} от {s.created_at:%d.%m.%Y %H:%M}")
    c.drawString(40, H - 115, f"Врач: {current_user.full_name}")

    y = H - 150
    for img in s.images:
        if y < 260:
            c.showPage()
            y = H - 80

        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, f"Снимок: {img.filename}")
        y -= 15

        c.setFont("Helvetica", 10)
        if not img.qc_ok:
            reasons = json.loads(img.qc_reasons or "[]")
            c.drawString(40, y, "QC: отклонён. " + "; ".join(reasons)[:120])
            y -= 15
            continue

        c.drawString(40, y, f"Диагноз модели: {CLASS_NAMES_RU.get(img.predicted_class, img.predicted_class)}")
        c.drawString(300, y, f"Уверенность: {img.confidence*100:.1f}%")
        y -= 15
        if img.bbox:
            c.drawString(40, y, f"BBox: {img.bbox} | Площадь: {img.area_pct}%")
            y -= 15

        try:
            c.drawImage(ImageReader(_abs(img.original_path)), 40, y - 170,
                        width=200, height=170, preserveAspectRatio=True)
            if img.overlay_path:
                c.drawImage(ImageReader(_abs(img.overlay_path)), 260, y - 170,
                            width=200, height=170, preserveAspectRatio=True)
            y -= 185
        except Exception:
            y -= 20

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 40, "Результат носит вспомогательный характер и не является медицинским заключением.")
    c.showPage()
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"neuroscan_session_{s.id}.pdf",
                     mimetype="application/pdf")


# ---------- bootstrap ----------
with app.app_context():
    db.create_all()
    # автосоздание админа, если БД пустая
    if not User.query.first():
        admin = User(
            username="admin",
            full_name="Администратор",
            role="admin",
            password_hash=generate_password_hash("admin123"),
        )
        db.session.add(admin)
        db.session.commit()
        print("[INIT] Создан admin/admin123 — смените пароль!")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)