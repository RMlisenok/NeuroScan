from datetime import datetime
import os, uuid, json
import numpy as np
import cv2
from flask import render_template, request, redirect, url_for, flash, abort, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app import app, db, ensure_model, allowed_file, _rel, _run_inference
from models_db import (
    Patient, Visit, VisitStatus,
    NeurologistReport, RadiologistReport, MRISession, MRIImage
)
from ml import CLASS_NAMES_RU
from qc import check_image
from symptoms_match import analyze_complaints
import io
from flask import send_file
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVu", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))

@app.route('/visit/<int:vid>', strict_slashes=False)
@login_required
def visit_view(vid):
    print(f"DEBUG visit_view: user={current_user.username}, role={current_user.role}, "
          f"is_neuro={current_user.is_neurologist}, is_radio={current_user.is_radiologist}")
    visit = Visit.query.get_or_404(vid)
    # любой врач может смотреть
    if not (current_user.is_neurologist or current_user.is_radiologist):
        abort(403)
    return render_template(
        'visit_view.html',
        visit=visit,
        patient=visit.patient,
        primary=visit.primary_report(),
        radiology=visit.radiologist_report,
        followup=visit.followup_report(),
    )

@app.route('/visit/<int:vid>/primary', methods=['GET', 'POST'])
@login_required
def visit_primary(vid):
    if not current_user.is_neurologist:
        abort(403)
    visit = Visit.query.get_or_404(vid)
    report = visit.primary_report()

    symptom_result = None
    if visit.patient.complaints:
        symptom_result = analyze_complaints(visit.patient.complaints)

    if request.method == 'POST':
        if report is None:
            report = NeurologistReport(
                visit_id=visit.id,
                doctor_id=current_user.id,
                kind='primary',
            )
            db.session.add(report)

        report.complaints = request.form.get('complaints', '').strip()
        report.disease_anamnesis = request.form.get('disease_anamnesis', '').strip()
        report.life_anamnesis = request.form.get('life_anamnesis', '').strip()
        report.objective_status = request.form.get('objective_status', '').strip()
        report.preliminary_diagnosis = request.form.get('preliminary_diagnosis', '').strip()
        report.examination_plan = request.form.get('examination_plan', '').strip()

        # Обновляем жалобы и в карточке пациента (чтобы модель видела актуальные)
        visit.patient.complaints = report.complaints

        if request.form.get('send_to_mri'):
            if visit.mri_session is None:
                session = MRISession(patient_id=visit.patient_id, status='pending')
                db.session.add(session)
                db.session.flush()
                visit.mri_session_id = session.id
            visit.status = VisitStatus.AWAITING_MRI
        else:
            visit.status = VisitStatus.CLOSED

        db.session.commit()
        flash('Первичный приём сохранён.', 'success')
        return redirect(url_for('patient_card', pid=visit.patient_id))

    return render_template(
        'visit_primary.html',
        visit=visit,
        report=report,
        patient=visit.patient,
        symptom_result=symptom_result,
        CLASS_NAMES_RU=CLASS_NAMES_RU,
    )


@app.route('/visit/<int:vid>/radiology', methods=['GET', 'POST'])
@login_required
def visit_radiology(vid):
    if not current_user.is_radiologist:
        abort(403)

    visit = Visit.query.get_or_404(vid)
    if visit.status != VisitStatus.AWAITING_MRI:
        flash('Этот визит не ожидает МРТ.', 'warning')
        return redirect(url_for('visits_list'))

    report = visit.radiologist_report

    if request.method == 'POST':
        action = request.form.get('action', '')

        # ===== 1. Загрузка снимков =====
        if action == 'upload':
            files = request.files.getlist('files')
            files = [f for f in files if f and f.filename]

            if not files:
                flash('Не выбрано ни одного файла.', 'warning')
                return redirect(request.url)

            # from app import ensure_model, _run_inference, _rel, allowed_file
            # from qc import check_image
            # import os, uuid, json, numpy as np, cv2
            # from werkzeug.utils import secure_filename
            # from flask import current_app

            if not ensure_model():
                flash('Модель не загружена.', 'danger')
                return redirect(request.url)

            s = visit.mri_session
            if s is None:
                s = MRISession(patient_id=visit.patient_id, status='pending')
                db.session.add(s)
                db.session.flush()
                visit.mri_session_id = s.id

            folder = os.path.join(current_app.config["UPLOAD_FOLDER"], f"p{visit.patient_id}", f"s{s.id}")
            os.makedirs(folder, exist_ok=True)

            added = 0
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
                added += 1

            db.session.commit()
            flash(f'Загружено снимков: {added}.', 'success')
            return redirect(url_for('visit_radiology', vid=visit.id))

        # ===== 2. Сохранение заключения =====
        if action == 'save_report':
            if report is None:
                report = RadiologistReport(
                    visit_id=visit.id,
                    mri_session_id=visit.mri_session_id,
                    radiologist_id=current_user.id,
                )
                db.session.add(report)

            report.area = request.form.get('area', '').strip()
            report.purpose = request.form.get('purpose', '').strip()
            report.description = request.form.get('description', '').strip()
            report.conclusion = request.form.get('conclusion', '').strip()
            report.signed_at = datetime.utcnow()

            visit.status = VisitStatus.AWAITING_FOLLOWUP
            db.session.commit()
            flash('Заключение подписано.', 'success')
            return redirect(url_for('visits_list'))

    return render_template(
        'visit_radiology.html',
        visit=visit,
        report=report,
        patient=visit.patient,
        config=current_app.config,
    )


@app.route('/visit/<int:vid>/followup', methods=['GET', 'POST'])
@login_required
def visit_followup(vid):
    if not current_user.is_neurologist:
        abort(403)
    visit = Visit.query.get_or_404(vid)
    if visit.status != VisitStatus.AWAITING_FOLLOWUP:
        flash('Этот визит не ожидает повторного приёма.', 'warning')
        return redirect(url_for('dashboard'))
    report = visit.followup_report()
    images = []
    if visit.mri_session:
        images = MRIImage.query.filter_by(session_id=visit.mri_session.id).all()

    if request.method == 'POST':
        if report is None:
            report = NeurologistReport(visit_id=visit.id, doctor_id=current_user.id, kind='followup')
            db.session.add(report)

        report.treatment = request.form.get('treatment', '').strip()
        report.prescribed_tests = request.form.get('prescribed_tests', '').strip()
        report.followup_notes = request.form.get('followup_notes', '').strip()
        report.diagnosis = request.form.get('diagnosis', '').strip()

        visit.status = VisitStatus.CLOSED
        visit.closed_at = datetime.utcnow()
        db.session.commit()
        flash('Приём завершён.', 'success')
        return redirect(url_for('patient_card', pid=visit.patient_id))

    return render_template(
        'visit_followup.html',
        visit=visit,
        report=report,
        patient=visit.patient,
        images=images,
    )


@app.route('/visits/processed')
@login_required
def visits_processed():
    if current_user.is_radiologist:
        visits = Visit.query.filter(
            Visit.status.in_([VisitStatus.AWAITING_FOLLOWUP, VisitStatus.CLOSED])
        ).order_by(Visit.created_at.desc()).all()
    else:
        visits = Visit.query.filter_by(status=VisitStatus.CLOSED) \
            .order_by(Visit.created_at.desc()).all()
    return render_template('visits_processed.html', visits=visits)

@app.route('/visits')
@login_required
def visits_list():
    if current_user.is_radiologist:
        visits = Visit.query.filter_by(status=VisitStatus.AWAITING_MRI)\
                            .order_by(Visit.created_at.desc()).all()
    else:
        visits = Visit.query.filter(
            Visit.status.in_([VisitStatus.OPEN, VisitStatus.AWAITING_FOLLOWUP])
        ).order_by(Visit.created_at.desc()).all()
    return render_template('visits_list.html', visits=visits)


# def _wrap(text, width):
#     words = (text or "").split()
#     lines, cur = [], ""
#     for w in words:
#         if len(cur) + len(w) + 1 <= width:
#             cur = (cur + " " + w).strip()
#         else:
#             if cur: lines.append(cur)
#             cur = w
#     if cur: lines.append(cur)
#     return lines or [""]


@app.route('/visit/<int:vid>/report.pdf')
@login_required
def visit_report(vid):
    visit = Visit.query.get_or_404(vid)
    from pdf_utils import build_primary_pdf, build_radiology_pdf, build_followup_pdf

    if visit.followup_report():
        return build_followup_pdf(visit)
    if visit.radiologist_report:
        return build_radiology_pdf(visit)
    if visit.primary_report():
        return build_primary_pdf(visit)
    abort(400)