"""
Инициализация БД + создание первого админа.
Запуск: python init_db.py
"""

import os
import getpass
from datetime import date

from app import app
from models_db import db, User, Patient, MRISession, MRIImage, AuditLog


def create_admin():
    print("\n=== Создание администратора ===")
    username = input("Логин: ").strip()
    if not username:
        print("Отмена.")
        return

    if User.query.filter_by(username=username).first():
        print(f"Пользователь '{username}' уже существует.")
        return

    full_name = input("ФИО: ").strip() or "Администратор"
    pw1 = getpass.getpass("Пароль: ")
    pw2 = getpass.getpass("Повторите пароль: ")
    if pw1 != pw2:
        print("Пароли не совпадают.")
        return
    if len(pw1) < 6:
        print("Пароль минимум 6 символов.")
        return

    u = User(username=username, full_name=full_name, role="admin")
    u.set_password(pw1)
    db.session.add(u)
    db.session.commit()
    print(f"✅ Админ '{username}' создан.")


def create_demo_doctor():
    """Опционально — демо-врач для показа."""
    if User.query.filter_by(username="doctor").first():
        return
    u = User(username="doctor", full_name="Иванов И.И.", role="doctor")
    u.set_password("doctor123")
    db.session.add(u)
    db.session.commit()
    print("✅ Демо-врач: doctor / doctor123")


def create_demo_patient():
    """Опционально — демо-пациент."""
    doctor = User.query.filter_by(username="doctor").first()
    if not doctor:
        return
    if Patient.query.filter_by(full_name="Демо Пациент").first():
        return
    p = Patient(
        doctor_id=doctor.id,
        full_name="Демо Пациент",
        birth_date=date(1985, 6, 15),
        gender="М",
        phone="+7 900 000-00-00",
        complaints="Упорная головная боль, тошнота, эпизоды судорог.",
        anamnesis="Ранее без особенностей. Наследственность не отягощена.",
    )
    db.session.add(p)
    db.session.commit()
    print("✅ Демо-пациент создан.")


def main():
    os.makedirs("instance", exist_ok=True)

    with app.app_context():
        print("Создание таблиц...")
        db.create_all()
        print("✅ Таблицы созданы:")
        for t in db.metadata.sorted_tables:
            print(f"   - {t.name}")

        if not User.query.first():
            create_admin()
        else:
            print(f"\nВ БД уже есть пользователи: {User.query.count()}")

        if input("\nСоздать демо-данных? (y/n): ").strip().lower() == "y":
            create_demo_doctor()
            create_demo_patient()

        print("\n=== Итог ===")
        print(f"Пользователей: {User.query.count()}")
        print(f"Пациентов:     {Patient.query.count()}")
        print(f"Сессий МРТ:    {MRISession.query.count()}")
        print(f"Снимков:       {MRIImage.query.count()}")


if __name__ == "__main__":
    main()