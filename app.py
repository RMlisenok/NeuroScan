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
from models_db import db, User, Patient, MRISession, MRIImage, AuditLog, Visit, VisitStatus, NeurologistReport, RadiologistReport
from config import Config
from models_db import db, User, Patient, MRISession, MRIImage, AuditLog
from ml import (build_model, GradCAM, prepare_input, overlay_cam,
                largest_bbox, bbox_area_pct, CLASS_NAMES_RU)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from class_info import CLASS_INFO
from symptoms_match import analyze_complaints
from pdf_utils import (
    build_session_pdf,
    build_radiology_pdf,
    build_primary_pdf,
    build_followup_pdf,
)

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


ALLOWED_EXTENSIONS = {'nii', 'nii.gz', 'dcm', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    if current_user.is_admin:
        return p
    if current_user.is_neurologist and p.doctor_id == current_user.id:
        return p
    if current_user.is_radiologist:
        # рентгенолог может смотреть любого пациента,
        # у которого есть визит в статусе awaiting_mri или awaiting_followup
        has_access = any(v.status in ('awaiting_mri', 'awaiting_followup') for v in p.visits)
        if has_access:
            return p
    abort(403)


def _get_session_or_404(sid):
    s = MRISession.query.get_or_404(sid)
    if current_user.is_admin:
        return s
    if current_user.is_radiologist:
        return s   # рентгенолог видит все сессии
    if s.patient.doctor_id == current_user.id:
        return s
    abort(403)


def _get_image_or_404(iid):
    img = MRIImage.query.get_or_404(iid)
    if current_user.is_admin or current_user.is_radiologist:
        return img
    if img.session.patient.doctor_id == current_user.id:
        return img
    abort(403)


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
import routes_visits

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)