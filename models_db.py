"""
NeuroScan AI — модели базы данных.

Таблицы:
  users         — врачи и админы
  patients      — пациенты
  mri_sessions  — сессии загрузки МРТ
  mri_images    — отдельные снимки
  audit_log     — журнал действий
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


# ============================================================
# USERS
# ============================================================
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(32), default="doctor", nullable=False)  # doctor | admin
    password_hash = db.Column(db.String(256), nullable=False)
    is_active_flag = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)

    patients = db.relationship(
        "Patient", backref="doctor", lazy="dynamic",
        cascade="all, delete-orphan"
    )

    def set_password(self, pw: str):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self.password_hash, pw)

    @property
    def is_active(self):  # flask-login
        return self.is_active_flag

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ============================================================
# PATIENTS
# ============================================================
class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)

    # Личные данные
    full_name = db.Column(db.String(128), nullable=False, index=True)
    birth_date = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(16), nullable=True)  # М / Ж
    phone = db.Column(db.String(32), nullable=True)
    email = db.Column(db.String(128), nullable=True)
    address = db.Column(db.String(256), nullable=True)
    insurance_number = db.Column(db.String(64), nullable=True)

    # Медицинские данные
    anamnesis = db.Column(db.Text, nullable=True)      # анамнез
    complaints = db.Column(db.Text, nullable=True)     # жалобы
    notes = db.Column(db.Text, nullable=True)          # заметки врача

    # Метаданные
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow, nullable=False)

    sessions = db.relationship(
        "MRISession", backref="patient", lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="MRISession.created_at.desc()"
    )

    @property
    def age(self):
        if not self.birth_date:
            return None
        today = datetime.utcnow().date()
        return today.year - self.birth_date.year - (
            (today.month, today.day) < (self.birth_date.month, self.birth_date.day)
        )

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "age": self.age,
            "gender": self.gender,
            "phone": self.phone,
            "email": self.email,
            "sessions_count": self.sessions.count(),
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<Patient {self.id} {self.full_name}>"


# ============================================================
# MRI SESSIONS
# ============================================================
class MRISession(db.Model):
    __tablename__ = "mri_sessions"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer,
                           db.ForeignKey("patients.id", ondelete="CASCADE"),
                           nullable=False, index=True)

    status = db.Column(db.String(32), default="uploaded", nullable=False)
    # uploaded | analyzed | edited | failed

    summary = db.Column(db.Text, nullable=True)          # агрегированный вывод
    final_diagnosis = db.Column(db.String(64), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow, nullable=False)

    images = db.relationship(
        "MRIImage", backref="session", lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="MRIImage.id"
    )

    @property
    def images_count(self):
        return self.images.count()

    @property
    def qc_passed_count(self):
        return self.images.filter_by(qc_ok=True).count()

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "status": self.status,
            "images_count": self.images_count,
            "qc_passed": self.qc_passed_count,
            "final_diagnosis": self.final_diagnosis,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<MRISession {self.id} patient={self.patient_id} status={self.status}>"


# ============================================================
# MRI IMAGES
# ============================================================
class MRIImage(db.Model):
    __tablename__ = "mri_images"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer,
                           db.ForeignKey("mri_sessions.id", ondelete="CASCADE"),
                           nullable=False, index=True)

    # Файлы
    filename = db.Column(db.String(256), nullable=False)          # оригинальное имя
    original_path = db.Column(db.String(512), nullable=False)     # относительный путь
    overlay_path = db.Column(db.String(512), nullable=True)       # Grad-CAM
    edited_mask_path = db.Column(db.String(512), nullable=True)   # правка врача

    # QC
    qc_ok = db.Column(db.Boolean, default=False, nullable=False)
    qc_reasons = db.Column(db.Text, nullable=True)     # JSON-массив
    qc_metrics = db.Column(db.Text, nullable=True)     # JSON-объект

    # Предсказание
    predicted_class = db.Column(db.String(64), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    probabilities = db.Column(db.Text, nullable=True)  # JSON {class: prob}
    bbox = db.Column(db.String(64), nullable=True)     # "x,y,w,h"
    area_pct = db.Column(db.Float, nullable=True)
    cam_threshold = db.Column(db.Float, default=0.5)

    # Флаги
    is_sent_to_training = db.Column(db.Boolean, default=False, nullable=False)
    doctor_verified = db.Column(db.Boolean, default=False, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "qc_ok": self.qc_ok,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<MRIImage {self.id} {self.filename} qc={self.qc_ok}>"


# ============================================================
# AUDIT LOG
# ============================================================
class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"),
                        nullable=True, index=True)
    action = db.Column(db.String(64), nullable=False, index=True)
    entity = db.Column(db.String(64), nullable=True)
    entity_id = db.Column(db.Integer, nullable=True)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow,
                           nullable=False, index=True)

    user = db.relationship("User", backref="logs")

    def __repr__(self):
        return f"<AuditLog {self.action} user={self.user_id}>"