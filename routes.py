import os
import uuid
import json
import numpy as np
import cv2
from flask import (request, redirect, url_for, flash,
                   render_template, abort, send_file, jsonify, session)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user, login_user, logout_user
from app import app, db, log_action, _parse_date, _get_patient_or_404, _get_session_or_404, _get_image_or_404, \
    ensure_model, _rel, _abs, _run_inference, _build_pdf
from models_db import MRISession, MRIImage, User, Patient, AuditLog
from auth_utils import admin_required, doctor_or_admin_required, doctor_required
from qc import check_image
from ml import CLASS_NAMES_RU
from symptoms_match import analyze_complaints

ALLOWED_EXTENSIONS = {'nii', 'nii.gz', 'dcm', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/session/<int:session_id>/upload', methods=['POST'])
def upload_mri(session_id):
    session = MRISession.query.get_or_404(session_id)

    if 'file' not in request.files:
        flash('Файл не выбран')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('Имя файла пустое')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Недопустимый формат файла')
        return redirect(request.url)

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    mri_image = MRIImage(session_id=session.id, file_path=unique_name)
    db.session.add(mri_image)
    db.session.commit()

    flash('МРТ загружен')
    return redirect(url_for('view_mri', image_id=mri_image.id))


# @app.route('/mri/<int:image_id>')
# def view_mri(image_id):
#     mri = MRIImage.query.get_or_404(image_id)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], mri.file_path)
#
#     if not os.path.exists(filepath):
#         abort(404)
#
#     predicted_class, confidence = predict_mri(filepath)
#
#     return render_template(
#         'mri_view.html',
#         mri=mri,
#         predicted_class=predicted_class,
#         confidence=round(confidence * 100, 2)
#     )

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


@app.route("/")
@login_required
def dashboard():
    q = request.args.get("q", "").strip()
    query = Patient.query.filter_by(doctor_id=current_user.id)
    if q:
        query = query.filter(Patient.full_name.ilike(f"%{q}%"))
    patients = query.order_by(Patient.updated_at.desc()).all()
    return render_template("dashboard.html", patients=patients, q=q)

@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin_users.html", users=users)


@app.route("/admin/users/<int:uid>/role", methods=["POST"])
@login_required
@admin_required
def admin_change_role(uid):
    u = User.query.get_or_404(uid)
    new_role = request.form.get("role")

    if new_role not in ("doctor", "admin"):
        abort(400)
    if u.id == current_user.id:
        flash("Нельзя менять свою роль")
        return redirect(url_for("admin_users"))

    u.role = new_role
    db.session.commit()
    log_action("role_change", "user", u.id, f"new_role={new_role}")
    flash(f"Роль {u.username} изменена на {new_role}")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:uid>/toggle", methods=["POST"])
@login_required
@admin_required
def admin_toggle_user(uid):
    u = User.query.get_or_404(uid)
    if u.id == current_user.id:
        flash("Нельзя деактивировать себя")
        return redirect(url_for("admin_users"))

    u.is_active_flag = not u.is_active_flag
    db.session.commit()
    log_action("user_toggle", "user", u.id, f"active={u.is_active_flag}")
    flash(f"Пользователь {u.username} "
          f"{'активирован' if u.is_active_flag else 'деактивирован'}")
    return redirect(url_for("admin_users"))


@app.route("/admin/logs")
@login_required
@admin_required
def admin_logs():
    """Просмотр журнала действий."""
    page = request.args.get("page", 1, type=int)
    per_page = 50
    pagination = (AuditLog.query
                  .order_by(AuditLog.created_at.desc())
                  .paginate(page=page, per_page=per_page, error_out=False))
    return render_template("admin_logs.html",
                           logs=pagination.items,
                           pagination=pagination)

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
    first_img = images[0] if images else None
    predicted = first_img.predicted_class if first_img else None
    symptom_result = analyze_complaints(s.patient.complaints or "", predicted_class=predicted)

    return render_template(
        "mri_result.html",
        session=s,
        patient=s.patient,
        images=images,
        symptom_result=symptom_result,
    )

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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

