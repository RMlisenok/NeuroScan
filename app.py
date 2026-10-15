import os
import io
import json
import uuid
import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, send_file, jsonify, abort, session)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from models_db import db, User, Patient, MRISession, MRIImage, AuditLog
from ml import (build_model, GradCAM, prepare_input, overlay_cam,
                largest_bbox, bbox_area_pct, CLASS_NAMES_RU)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from class_info import CLASS_INFO
from symptoms_match import analyze_complaints

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVu", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", f"{FONT_DIR}/DejaVuSans-Oblique.ttf"))


load_dotenv(Path(__file__).resolve().parent / '.env')
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("instance", exist_ok=True)
os.makedirs("models", exist_ok=True)

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

@app.template_filter("ru_class")
def ru_class_filter(class_name):
    if not class_name:
        return "—"
    return CLASS_NAMES_RU.get(class_name, class_name)

@app.template_filter("ru_class")
def ru_class_filter(class_name):
    """glioma -> Глиома"""
    if not class_name:
        return "—"
    return CLASS_NAMES_RU.get(class_name, class_name)


@app.template_filter("from_json")
def from_json_filter(value):
    """JSON-строка -> dict/list. Безопасно возвращает пустой объект при ошибке."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}



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


def _wrap_text(text, width):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _build_pdf(s):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from datetime import datetime as _dt
    FONT_DIR = "/usr/share/fonts/truetype/dejavu"
    pdfmetrics.registerFont(TTFont("DejaVu", f"{FONT_DIR}/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", f"{FONT_DIR}/DejaVuSans-Oblique.ttf"))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    c.setFont("DejaVu-Bold", 16)
    c.drawString(40, H - 50, "NeuroScan AI — отчёт по МРТ-исследованию")

    c.setFont("DejaVu", 10)
    y = H - 80

    doctor = s.patient.doctor
    doctor_name = doctor.full_name if doctor else "—"
    department = getattr(doctor, "department", None) or "—"

    age = s.patient.age if hasattr(s.patient, "age") else "—"

    gender_raw = s.patient.gender or "—"
    gender = {"М": "Мужской", "Ж": "Женский", "M": "Мужской", "F": "Женский"}.get(gender_raw, gender_raw)

    exam_time = s.created_at.strftime("%d.%m.%Y %H:%M") if s.created_at else "—"
    first_img = s.images.first() if hasattr(s.images, "first") else (list(s.images)[0] if s.images else None)

    diagnosis = "—"
    if first_img and first_img.predicted_class:
        from ml import CLASS_NAMES_RU
        diagnosis = CLASS_NAMES_RU.get(first_img.predicted_class, first_img.predicted_class)
    lines = [
        ("ФИО пациента:", s.patient.full_name or "—"),
        ("Возраст:", f"{age} лет" if age != "—" else "—"),
        ("Пол:", gender),
        ("Лечащий врач:", doctor_name),
        ("Отделение:", department),
        ("Тип обследования:", "МРТ головного мозга"),
        ("Время обследования:", exam_time),
        ("Диагноз (предварительный):", diagnosis),
    ]

    c.setFont("DejaVu", 10)
    for label, value in lines:
        c.setFont("DejaVu-Bold", 10)
        c.drawString(40, y, label)
        c.setFont("DejaVu", 10)
        c.drawString(200, y, str(value)[:80])
        y -= 16
    y -= 6
    c.setFont("DejaVu-Bold", 11)
    c.drawString(40, y, "Анамнез:")
    y -= 14
    c.setFont("DejaVu", 10)
    anamnesis = (s.patient.anamnesis or "Не указан").strip()
    for line in _wrap_text(anamnesis, 90):
        if y < 260:
            c.showPage()
            y = H - 60
            c.setFont("DejaVu", 10)
        c.drawString(40, y, line)
        y -= 13

    y -= 10
    c.setFont("DejaVu-Bold", 12)
    c.drawString(40, y, "Результаты анализа снимков:")
    y -= 20

    for i, img in enumerate(s.images, 1):
        if y < 300:
            c.showPage()
            y = H - 60

        c.setFont("DejaVu-Bold", 10)
        c.drawString(40, y, f"Снимок {i}: {img.filename}")
        y -= 14

        c.setFont("DejaVu", 9)
        if not img.qc_ok:
            try:
                reasons = json.loads(img.qc_reasons or "[]")
            except Exception:
                reasons = []
            c.drawString(40, y, "QC: отклонён. " + "; ".join(reasons)[:110])
            y -= 14
            continue

        from ml import CLASS_NAMES_RU
        cls_ru = CLASS_NAMES_RU.get(img.predicted_class, img.predicted_class or "—")
        c.drawString(40, y, f"Диагноз модели: {cls_ru}")
        if img.confidence is not None:
            c.drawString(320, y, f"Уверенность: {img.confidence*100:.1f}%")
        y -= 14

        if img.bbox:
            c.drawString(40, y, f"Область интереса (bbox): {img.bbox} | Площадь: {img.area_pct}%")
            y -= 14

        try:
            orig = _abs(img.original_path)
            c.drawImage(ImageReader(orig), 40, y - 160,
                        width=220, height=160, preserveAspectRatio=True)
            c.setFont("DejaVu", 8)
            c.drawString(40, y - 172, "Оригинал")
            if img.overlay_path:
                ov = _abs(img.overlay_path)
                c.drawImage(ImageReader(ov), 290, y - 160,
                            width=220, height=160, preserveAspectRatio=True)
                c.drawString(290, y - 172, "Grad-CAM (область внимания модели)")
            y -= 190
        except Exception:
            y -= 20

    if y < 200:
        c.showPage()
        y = H - 60
        
    from symptoms_match import analyze_complaints

    complaints = s.patient.complaints or ""
    first_img_for_pdf = s.images.first() if hasattr(s.images, "first") else None
    predicted_for_pdf = first_img_for_pdf.predicted_class if first_img_for_pdf else None
    sym = analyze_complaints(complaints, predicted_class=predicted_for_pdf)

    if sym["has_text"]:
        if y < 250:
            c.showPage()
            y = H - 60

        c.setFont("DejaVu-Bold", 12)
        c.drawString(40, y, "Сопоставление жалоб и результатов анализа")
        y -= 20

        c.setFont("DejaVu", 10)
        if sym["warning"]:
            c.setFillColorRGB(0.8, 0.4, 0)
            for line in _wrap_text("⚠ " + sym["warning"], 95):
                c.drawString(40, y, line)
                y -= 13
            c.setFillColorRGB(0, 0, 0)
        else:
            c.setFillColorRGB(0.1, 0.5, 0.1)
            c.drawString(40, y, "✅ Жалобы пациента согласуются с результатом модели.")
            c.setFillColorRGB(0, 0, 0)
            y -= 16

        y -= 6
        c.setFont("DejaVu-Bold", 10)
        c.drawString(40, y, "Класс")
        c.drawString(300, y, "Сходство с жалобами")
        y -= 14

        c.setFont("DejaVu", 10)
        for cls, score in sym["similarity"].items():
            name = CLASS_INFO.get(cls, {}).get("name_ru", cls)
            c.drawString(40, y, name)
            c.drawString(300, y, f"{score * 100:.1f}%")
            y -= 13
    y -= 10
    c.setFont("DejaVu", 10)
    now = _dt.now().strftime("%d.%m.%Y %H:%M")

    c.drawString(40, y, f"Выполнил (рентгенолог): ______________________  "
                        f"Время сбора данных: {now}")
    y -= 20
    c.drawString(40, y, f"Проверил (лечащий врач): {doctor_name}  "
                        f"Время проверки: ______________")
    y -= 24

    c.setFont("DejaVu", 10)
    c.drawString(40, y, "Подпись врача: ______________________")
    y -= 24

    c.setFont("DejaVu-Oblique", 9)
    note = ("Примечание: результат автоматического анализа носит вспомогательный характер "
            "и не является медицинским заключением. Окончательный диагноз устанавливает "
            "лечащий врач на основании клинической картины и дополнительных исследований.")
    for line in _wrap_text(note, 100):
        c.drawString(40, y, line)
        y -= 12

    c.showPage()
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"neuroscan_session_{s.id}.pdf",
                     mimetype="application/pdf")


with app.app_context():
    db.create_all()
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

import routes

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)