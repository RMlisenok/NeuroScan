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


# USERS
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
    ROLE_ADMIN = 'admin'
    ROLE_NEUROLOGIST = 'neurologist'
    ROLE_RADIOLOGIST = 'radiologist'

    @property
    def is_admin(self):
        return self.role == self.ROLE_ADMIN

    @property
    def is_neurologist(self):
        return self.role in (self.ROLE_NEUROLOGIST, self.ROLE_ADMIN)

    @property
    def is_radiologist(self):
        return self.role in (self.ROLE_RADIOLOGIST, self.ROLE_ADMIN)

    def set_password(self, pw: str):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self.password_hash, pw)

    @property
    def is_active(self):  # flask-login
        return self.is_active_flag

    def has_role(self, *roles) -> bool:
        return self.role in roles

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# PATIENTS
class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)

    # Личные данные
    full_name = db.Column(db.String(128), nullable=False, index=True)
    birth_date = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(16), nullable=True)
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
            "sessions_count": self.sessions.count(),
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<Patient {self.id} {self.full_name}>"


# MRI SESSIONS
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
        "MRIImage", back_populates="session", lazy="dynamic",
        cascade="all, delete-orphan", order_by="MRIImage.id"
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


# MRI IMAGES
class MRIImage(db.Model):
    __tablename__ = 'mri_images'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('mri_sessions.id'), nullable=False)

    session = db.relationship('MRISession', back_populates='images')
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
    verified_class = db.Column(db.String(64), nullable=True)
    training_used = db.Column(db.Boolean, default=False, nullable=False)

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


# AUDIT LOG
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

from datetime import datetime

class VisitStatus:
    OPEN = 'open'
    AWAITING_MRI = 'awaiting_mri'
    AWAITING_FOLLOWUP = 'awaiting_followup'
    CLOSED = 'closed'

class Visit(db.Model):
    __tablename__ = 'visits'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    visit_type = db.Column(db.String(20), nullable=False, default='primary')

    status = db.Column(db.String(30), nullable=False, default='open')

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime, nullable=True)

    patient = db.relationship('Patient', backref=db.backref('visits', lazy='dynamic'))
    doctor = db.relationship('User', backref=db.backref('visits', lazy='dynamic'))

    mri_session_id = db.Column(db.Integer, db.ForeignKey('mri_sessions.id'), nullable=True)
    mri_session = db.relationship('MRISession', backref='visit', uselist=False)

    neuro_reports = db.relationship(
        'NeurologistReport', backref='visit', lazy='dynamic',
        cascade='all, delete-orphan'
    )
    radiologist_report = db.relationship(
        'RadiologistReport', backref='visit', uselist=False,
        cascade='all, delete-orphan'
    )

    def primary_report(self):
        return self.neuro_reports.filter_by(kind='primary').first()

    def followup_report(self):
        return self.neuro_reports.filter_by(kind='followup').first()

    def __repr__(self):
        return f'<Visit #{self.id} patient={self.patient_id} status={self.status}>'


class NeurologistReport(db.Model):
    __tablename__ = 'neurologist_reports'

    id = db.Column(db.Integer, primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey('visits.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    kind = db.Column(db.String(20), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    complaints = db.Column(db.Text)
    disease_anamnesis = db.Column(db.Text)
    life_anamnesis = db.Column(db.Text)
    objective_status = db.Column(db.Text)
    preliminary_diagnosis = db.Column(db.Text)
    examination_plan = db.Column(db.Text)
    diagnosis = db.Column(db.Text)

    treatment = db.Column(db.Text)
    prescribed_tests = db.Column(db.Text)
    followup_notes = db.Column(db.Text)

    doctor = db.relationship('User', backref='neuro_reports')

    def __repr__(self):
        return f'<NeurologistReport #{self.id} visit={self.visit_id} kind={self.kind}>'


class RadiologistReport(db.Model):
    __tablename__ = 'radiologist_reports'

    id = db.Column(db.Integer, primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey('visits.id'), nullable=False)
    mri_session_id = db.Column(db.Integer, db.ForeignKey('mri_sessions.id'), nullable=False)
    radiologist_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    signed_at = db.Column(db.DateTime, nullable=True)

    area = db.Column(db.String(100))
    purpose = db.Column(db.Text)
    description = db.Column(db.Text)
    conclusion = db.Column(db.Text)

    radiologist = db.relationship('User', backref='radiology_reports')

    def __repr__(self):
        return f'<RadiologistReport #{self.id} visit={self.visit_id}>'