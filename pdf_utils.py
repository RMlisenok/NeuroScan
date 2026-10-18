import os
import io
import json
from datetime import datetime

from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from ml import CLASS_NAMES_RU
from class_info import CLASS_INFO

# ---------- Шрифты ----------
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVu", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", f"{FONT_DIR}/DejaVuSans-Oblique.ttf"))

APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- Утилиты ----------
def _abs(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(APP_DIR, path)


def _wrap(text, width):
    words = (text or "").split()
    lines, cur = [], ""
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


def _draw_header(c, W, H, title):
    logo = os.path.join(APP_DIR, "static", "logo.png")
    try:
        c.drawImage(ImageReader(logo), 40, H - 70, width=32, height=32,
                    preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    c.setFont("DejaVu-Bold", 18)
    c.drawString(80, H - 55, "LUMINA")

    c.setLineWidth(0.5)
    c.line(40, H - 75, W - 40, H - 75)

    c.setFont("DejaVu-Bold", 14)
    c.drawString(40, H - 100, title)
    return H - 125


def _draw_fields(c, y, fields, label_x=40, value_x=230, W=None, H=None):
    c.setFont("DejaVu", 10)
    for label, value in fields:
        c.setFont("DejaVu-Bold", 10)
        c.drawString(label_x, y, label)
        c.setFont("DejaVu", 10)
        text = str(value or "—")

        if len(text) <= 55:
            # короткое — в одну строку рядом с меткой
            c.drawString(value_x, y, text)
            y -= 16
        else:
            # длинное — с новой строки под меткой
            y -= 14
            for line in _wrap(text, 90):
                if H and y < 80:
                    c.showPage()
                    y = _draw_header(c, W, H, "")
                c.drawString(label_x + 10, y, line)
                y -= 12
            y -= 4
    return y

def _draw_block(c, y, title, text, W, H):
    c.setFont("DejaVu-Bold", 11)
    c.drawString(40, y, title)
    y -= 14
    c.setFont("DejaVu", 10)
    for line in _wrap(text or "—", 95):
        if y < 80:
            c.showPage()
            y = _draw_header(c, W, H, "")
            c.setFont("DejaVu", 10)
        c.drawString(50, y, line)
        y -= 13
    return y - 6


def _gender_ru(g):
    return {"М": "Мужской", "Ж": "Женский", "M": "Male", "F": "Female"}.get(g or "", g or "—")


def _safe_age(p):
    try:
        return p.age
    except Exception:
        return "—"


def _send(buf, name):
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=name, mimetype="application/pdf")


# ---------- 1. Заключение рентгенолога ----------
def build_radiology_pdf(visit):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = _draw_header(c, W, H, "Заключение рентгенолога")

    rad = visit.radiologist_report
    patient = visit.patient

    fields = [
        ("ФИО пациента:", patient.full_name or "—"),
        ("Дата рождения:", str(patient.birth_date or "—")),
        ("Пол:", _gender_ru(patient.gender)),
        ("Полис:", patient.insurance_number or "—"),
        ("Дата исследования:", visit.created_at.strftime("%d.%m.%Y %H:%M")),
        ("Специалист:", rad.radiologist.full_name if rad and rad.radiologist else "—"),
    ]
    y = _draw_fields(c, y, fields, W=W, H=H)
    y -= 10

    y = _draw_block(c, y, "Область исследования:", rad.area if rad else "—", W, H)
    y = _draw_block(c, y, "Цель исследования:", rad.purpose if rad else "—", W, H)
    y = _draw_block(c, y, "Описание:", rad.description if rad else "—", W, H)
    y = _draw_block(c, y, "Заключение:", rad.conclusion if rad else "—", W, H)

    c.setFont("DejaVu-Oblique", 9)
    for line in _wrap("* Заключение не является диагнозом и должно быть интерпретировано врачом.", 100):
        c.drawString(40, y, line)
        y -= 12

    c.showPage()
    c.save()
    return _send(buf, f"lumina_radiology_{visit.id}.pdf")


# ---------- 2. Первичный приём невролога ----------
def build_primary_pdf(visit):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = _draw_header(c, W, H, "Запись врача невролога. Первичный приём")

    rep = visit.primary_report()
    patient = visit.patient

    fields = [
        ("Дата приёма:", visit.created_at.strftime("%d.%m.%Y %H:%M")),
        ("Специалист:", rep.doctor.full_name if rep and rep.doctor else "—"),
        ("ФИО пациента:", patient.full_name or "—"),
        ("Дата рождения:", str(patient.birth_date or "—")),
        ("Пол:", _gender_ru(patient.gender)),
        ("Возраст:", f"{_safe_age(patient)} лет"),
        ("Полис:", patient.insurance_number or "—"),
    ]
    y = _draw_fields(c, y, fields, W=W, H=H)
    y -= 10

    y = _draw_block(c, y, "Жалобы:", rep.complaints if rep else "—", W, H)
    y = _draw_block(c, y, "Анамнез настоящего заболевания:", rep.disease_anamnesis if rep else "—", W, H)
    y = _draw_block(c, y, "Анамнез жизни:", rep.life_anamnesis if rep else "—", W, H)
    y = _draw_block(c, y, "Объективный статус:", rep.objective_status if rep else "—", W, H)
    y = _draw_block(c, y, "Диагноз (предположительный):", rep.preliminary_diagnosis if rep else "—", W, H)
    y = _draw_block(c, y, "План обследований и действий:", rep.examination_plan if rep else "—", W, H)

    c.setFont("DejaVu", 10)
    c.drawString(40, y, "Подпись врача: ______________________")
    c.showPage()
    c.save()
    return _send(buf, f"lumina_primary_{visit.id}.pdf")


# ---------- 3. Повторный приём невролога ----------
def build_followup_pdf(visit):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = _draw_header(c, W, H, "Запись врача невролога. Повторный приём")

    rep = visit.followup_report()
    rad = visit.radiologist_report
    patient = visit.patient

    fields = [
        ("Дата приёма:", (visit.closed_at or visit.created_at).strftime("%d.%m.%Y %H:%M")),
        ("Специалист:", rep.doctor.full_name if rep and rep.doctor else "—"),
        ("ФИО пациента:", patient.full_name or "—"),
        ("Дата рождения:", str(patient.birth_date or "—")),
        ("Пол:", _gender_ru(patient.gender)),
        ("Возраст:", f"{_safe_age(patient)} лет"),
        ("Полис:", patient.insurance_number or "—"),
    ]
    y = _draw_fields(c, y, fields, W=W, H=H)
    y -= 10

    y = _draw_block(c, y, "Жалобы:", patient.complaints or "—", W, H)

    if rad:
        radiology_text = (
            f"Область: {rad.area or '—'}\n"
            f"Описание: {rad.description or '—'}\n"
            f"Заключение: {rad.conclusion or '—'}"
        )
        y = _draw_block(c, y, "Заключение рентгенолога:", radiology_text, W, H)

    # Диагноз — без «предположительный»
    y = _draw_block(c, y, "Диагноз:", rep.preliminary_diagnosis if rep else "—", W, H)

    # План лечения — объединяем treatment + prescribed_tests
    plan_parts = []
    if rep:
        if rep.treatment:
            plan_parts.append(rep.treatment.strip())
        if rep.prescribed_tests:
            plan_parts.append("Анализы: " + rep.prescribed_tests.strip())
    plan_text = "\n".join(plan_parts) or "—"
    y = _draw_block(c, y, "План лечения:", plan_text, W, H)

    # Заметки врача — НЕ выводим (только для врача)

    c.setFont("DejaVu", 10)
    c.drawString(40, y, "Подпись врача: ______________________")
    c.showPage()
    c.save()
    return _send(buf, f"lumina_followup_{visit.id}.pdf")


# ---------- 4. Отчёт по сессии МРТ (полный, со снимками) ----------
def build_session_pdf(session):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = _draw_header(c, W, H, "Отчёт по МРТ-исследованию")

    patient = session.patient
    doctor = patient.doctor
    first_img = session.images.first() if hasattr(session.images, "first") else (
        list(session.images)[0] if session.images else None)

    diagnosis = "—"
    if first_img and first_img.predicted_class:
        diagnosis = CLASS_NAMES_RU.get(first_img.predicted_class, first_img.predicted_class)

    fields = [
        ("ФИО пациента:", patient.full_name or "—"),
        ("Возраст:", f"{_safe_age(patient)} лет"),
        ("Пол:", _gender_ru(patient.gender)),
        ("Полис:", patient.insurance_number or "—"),
        ("Лечащий врач:", doctor.full_name if doctor else "—"),
        ("Тип обследования:", "МРТ головного мозга"),
        ("Время обследования:", session.created_at.strftime("%d.%m.%Y %H:%M")),
        ("Диагноз (предположительный):", diagnosis),
    ]
    y = _draw_fields(c, y, fields, W=W, H=H)
    y -= 10

    y = _draw_block(c, y, "Анамнез:", patient.anamnesis or "Не указан", W, H)

    c.setFont("DejaVu-Bold", 12)
    c.drawString(40, y, "Результаты анализа снимков:")
    y -= 20

    for i, img in enumerate(session.images, 1):
        if y < 300:
            c.showPage()
            y = _draw_header(c, W, H, "")
            c.setFont("DejaVu", 10)

        c.setFont("DejaVu-Bold", 10)
        c.drawString(40, y, f"Снимок {i}: {img.filename}")
        y -= 14

        if not img.qc_ok:
            try:
                reasons = json.loads(img.qc_reasons or "[]")
            except Exception:
                reasons = []
            c.setFont("DejaVu", 9)
            c.drawString(40, y, "QC: отклонён. " + "; ".join(reasons)[:110])
            y -= 14
            continue

        c.setFont("DejaVu", 9)
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

    # Сопоставление жалоб и модели
    from symptoms_match import analyze_complaints
    sym = analyze_complaints(patient.complaints or "",
                              predicted_class=first_img.predicted_class if first_img else None)
    if sym["has_text"]:
        if y < 250:
            c.showPage()
            y = _draw_header(c, W, H, "")
        c.setFont("DejaVu-Bold", 12)
        c.drawString(40, y, "Сопоставление жалоб и результатов анализа")
        y -= 20

        if sym["warning"]:
            c.setFillColorRGB(0.8, 0.4, 0)
            for line in _wrap("⚠ " + sym["warning"], 90):
                if y < 80:
                    c.showPage()
                    y = _draw_header(c, W, H, "")
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
    from datetime import datetime as _dt
    now_str = _dt.now().strftime("%d.%m.%Y %H:%M")

    rad_name = "—"
    rad_time = now_str
    if session.visit and session.visit.radiologist_report:
        rad = session.visit.radiologist_report
        if rad.radiologist:
            rad_name = rad.radiologist.full_name
        if rad.signed_at:
            rad_time = rad.signed_at.strftime("%d.%m.%Y %H:%M")

    doctor_name = doctor.full_name if doctor else "—"

    c.setFont("DejaVu", 10)
    c.drawString(40, y, f"Выполнил (рентгенолог): {rad_name}")
    y -= 16
    c.drawString(40, y, f"Время сбора данных: {rad_time}")
    y -= 16
    c.drawString(40, y, f"Проверил (лечащий врач): {doctor_name}")
    y -= 16
    c.drawString(40, y, "Время проверки: ______________________")
    y -= 16
    c.drawString(40, y, "Подпись врача: ______________________")
    y -= 20

    c.setFont("DejaVu-Oblique", 9)
    note = ("Примечание: результат автоматического анализа носит вспомогательный характер "
            "и не является медицинским заключением. Окончательный диагноз устанавливает "
            "лечащий врач на основании клинической картины и дополнительных исследований.")
    for line in _wrap(note, 100):
        c.drawString(40, y, line)
        y -= 12

    c.showPage()
    c.save()
    return _send(buf, f"lumina_session_{session.id}.pdf")