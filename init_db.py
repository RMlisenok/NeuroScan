import os
import getpass
from datetime import date, datetime

from app import app
from models_db import db, User, Patient, MRISession, MRIImage, AuditLog


def get_or_create_user(username, full_name, role, password):
    u = User.query.filter_by(username=username).first()
    if u:
        print(f"  ⏭  пользователь '{username}' уже есть")
        return u
    u = User(username=username, full_name=full_name, role=role)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    print(f"  ✅ создан '{username}' / {password} ({role})")
    return u

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

def create_demo_doctors():
    """Демо-невролог и демо-рентгенолог."""
    print("\n=== Демо-врачи ===")
    neurologist = get_or_create_user(
        username="neurologist",
        full_name="Иванов Иван Иванович",
        role="neurologist",
        password="neuro123",
    )
    radiologist = get_or_create_user(
        username="radiologist",
        full_name="Петрова Мария Сергеевна",
        role="radiologist",
        password="radio123",
    )
    return neurologist, radiologist

def create_demo_patient(neurologist):
    """Демо-пациент, привязанный к неврологу."""
    if not neurologist:
        print("  ⚠️  нет невролога — демо-пациент не создан")
        return

    if Patient.query.filter_by(full_name="Демо Пациент").first():
        print("  ⏭  демо-пациент уже есть")
        return

    p = Patient(
        doctor_id=neurologist.id,
        full_name="Демо Пациент",
        birth_date=date(1985, 6, 15),
        gender="М",
        complaints="Упорная головная боль, тошнота, эпизоды судорог.",
        anamnesis="Ранее без особенностей. Наследственность не отягощена.",
    )
    db.session.add(p)
    db.session.commit()
    print("  ✅ Демо-пациент создан (врач: neurologist)")


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

        if input("\nСоздать демо-данные? (y/n): ").strip().lower() == "y":
            neurologist, radiologist = create_demo_doctors()
            create_demo_patient(neurologist)

        print("\n=== Итог ===")
        print(f"Пользователей:   {User.query.count()}")
        print(f"  админов:       {User.query.filter_by(role='admin').count()}")
        print(f"  неврологов:    {User.query.filter_by(role='neurologist').count()}")
        print(f"  рентгенологов: {User.query.filter_by(role='radiologist').count()}")
        print(f"  прочих:        {User.query.filter_by(role='doctor').count()}")
        print(f"Пациентов:       {Patient.query.count()}")
        print(f"Сессий МРТ:      {MRISession.query.count()}")
        print(f"Снимков:         {MRIImage.query.count()}")


if __name__ == "__main__":
    main()