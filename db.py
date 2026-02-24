import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
from specialty import get_system_prompt

class MedicalAppointmentDB:
    def __init__(self, db_name="medical_appointments.db"):
        self.db_name = db_name
        self.init_database()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def init_database(self):
        """Initialize the database with tables for doctors, time slots, and appointments."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                specialty TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                is_available BOOLEAN DEFAULT 1,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                patient_phone TEXT NOT NULL,
                doctor_id INTEGER NOT NULL,
                time_slot_id INTEGER NOT NULL,
                status TEXT DEFAULT 'confirmed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id),
                FOREIGN KEY (time_slot_id) REFERENCES time_slots(id)
            )
        ''')

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM doctors")
        if cursor.fetchone()[0] == 0:
            self.seed_data(conn)

        conn.close()

    def seed_data(self, conn):
        """Seed diverse doctors and varied time slots across all specialties."""
        cursor = conn.cursor()

        # Two doctors per specialty, across all 22 specialties (44 doctors total)
        doctors = [
            # General
            ("Dr. Amelia Hart",         "General"),
            ("Dr. Samuel Osei",         "General"),
            # Radiology
            ("Dr. Fatima Al-Rashid",    "Radiology"),
            ("Dr. Liam Novak",          "Radiology"),
            # Pathology
            ("Dr. Priya Menon",         "Pathology"),
            ("Dr. Ethan Krause",        "Pathology"),
            # Dermatology
            ("Dr. Isabelle Fontaine",   "Dermatology"),
            ("Dr. Kwame Asante",        "Dermatology"),
            # Ophthalmology
            ("Dr. Yuki Tanaka",         "Ophthalmology"),
            ("Dr. Marcus Webb",         "Ophthalmology"),
            # Cardiology
            ("Dr. Sarah Johnson",       "Cardiology"),
            ("Dr. Rajan Patel",         "Cardiology"),
            # Oncology
            ("Dr. Helena Bergström",    "Oncology"),
            ("Dr. Taiwo Adeyemi",       "Oncology"),
            # Emergency
            ("Dr. Connor Reilly",       "Emergency"),
            ("Dr. Nadia Sousa",         "Emergency"),
            # Neurology
            ("Dr. Eleanor Marsh",       "Neurology"),
            ("Dr. Vikram Sharma",       "Neurology"),
            # Psychiatry
            ("Dr. Grace Liu",           "Psychiatry"),
            ("Dr. Omar Abdullah",       "Psychiatry"),
            # Orthopedics
            ("Dr. James Wilson",        "Orthopedics"),
            ("Dr. Sofía Herrera",       "Orthopedics"),
            # Pediatrics
            ("Dr. Emily Rodriguez",     "Pediatrics"),
            ("Dr. Tobias Müller",       "Pediatrics"),
            # Gynecology
            ("Dr. Aisha Kamara",        "Gynecology"),
            ("Dr. Charlotte Dupont",    "Gynecology"),
            # Urology
            ("Dr. Benjamin Adler",      "Urology"),
            ("Dr. Hiroshi Yamamoto",    "Urology"),
            # Gastroenterology
            ("Dr. Miriam Cohen",        "Gastroenterology"),
            ("Dr. Aleksei Volkov",      "Gastroenterology"),
            # Endocrinology
            ("Dr. Sunita Rao",          "Endocrinology"),
            ("Dr. Patrick O'Brien",     "Endocrinology"),
            # Pulmonology
            ("Dr. Leila Nazari",        "Pulmonology"),
            ("Dr. David Okonkwo",       "Pulmonology"),
            # Nephrology
            ("Dr. Anna Johansson",      "Nephrology"),
            ("Dr. Carlos Mendez",       "Nephrology"),
            # Rheumatology
            ("Dr. Ingrid Hansen",       "Rheumatology"),
            ("Dr. Femi Adesanya",       "Rheumatology"),
            # Hematology
            ("Dr. Rachel Goldstein",    "Hematology"),
            ("Dr. Takashi Watanabe",    "Hematology"),
            # Infectious Disease
            ("Dr. Amara Diallo",        "Infectious Disease"),
            ("Dr. Sebastian Koch",      "Infectious Disease"),
            # Anesthesiology
            ("Dr. Chloe Beaumont",      "Anesthesiology"),
            ("Dr. Ibrahim Hassan",      "Anesthesiology"),
        ]
        cursor.executemany("INSERT INTO doctors (name, specialty) VALUES (?, ?)", doctors)

        # Varied time slots: morning, mid-morning, midday, afternoon, late afternoon
        # Each doctor gets different slot patterns to create realistic variation
        all_time_slots = [
            ["08:00", "09:30", "11:00", "13:30", "15:00", "16:30"],  # Pattern A
            ["08:30", "10:00", "11:30", "14:00", "15:30", "17:00"],  # Pattern B
            ["09:00", "10:30", "12:00", "14:30", "16:00"],           # Pattern C
            ["07:30", "09:00", "10:30", "12:30", "14:00", "15:30"],  # Pattern D (early)
        ]

        start_date = datetime.now().date()
        # Generate 14 days of slots
        for doc_idx, (doc_name, _) in enumerate(doctors):
            doctor_id = doc_idx + 1
            slot_pattern = all_time_slots[doc_idx % len(all_time_slots)]
            # Each doctor works 5 of the next 14 days (staggered)
            working_days = [d for d in range(14) if (d + doc_idx) % 3 != 0][:10]
            for day_offset in working_days:
                date = start_date + timedelta(days=day_offset)
                for time_str in slot_pattern:
                    cursor.execute(
                        "INSERT INTO time_slots (doctor_id, date, time, is_available) VALUES (?, ?, ?, 1)",
                        (doctor_id, date, time_str)
                    )

        conn.commit()

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def check_availability(self, specialty=None, date=None):
        """Check available appointments."""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = '''
            SELECT d.name, d.specialty, ts.date, ts.time, ts.id
            FROM time_slots ts
            JOIN doctors d ON ts.doctor_id = d.id
            WHERE ts.is_available = 1
        '''
        params = []

        if specialty:
            query += " AND d.specialty LIKE ?"
            params.append(f"%{specialty}%")

        if date:
            query += " AND ts.date = ?"
            params.append(date)

        query += " ORDER BY ts.date, ts.time LIMIT 20"

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [
            {"doctor": r[0], "specialty": r[1], "date": r[2], "time": r[3], "slot_id": r[4]}
            for r in results
        ]

    def book_appointment(self, patient_name, patient_phone, time_slot_id):
        """Book a new appointment."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT is_available, doctor_id FROM time_slots WHERE id = ?", (time_slot_id,))
        result = cursor.fetchone()

        if not result or not result[0]:
            conn.close()
            return {"success": False, "message": "Time slot not available"}

        doctor_id = result[1]

        cursor.execute(
            "INSERT INTO appointments (patient_name, patient_phone, doctor_id, time_slot_id) VALUES (?, ?, ?, ?)",
            (patient_name, patient_phone, doctor_id, time_slot_id)
        )
        appointment_id = cursor.lastrowid

        cursor.execute("UPDATE time_slots SET is_available = 0 WHERE id = ?", (time_slot_id,))

        conn.commit()
        conn.close()

        return {"success": True, "appointment_id": appointment_id, "message": "Appointment booked successfully"}

    def get_appointment(self, appointment_id):
        """Get appointment details."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT a.id, a.patient_name, a.patient_phone, d.name, d.specialty,
                   ts.date, ts.time, a.status
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            JOIN time_slots ts ON a.time_slot_id = ts.id
            WHERE a.id = ?
        ''', (appointment_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "id": result[0],
                "patient_name": result[1],
                "patient_phone": result[2],
                "doctor": result[3],
                "specialty": result[4],
                "date": result[5],
                "time": result[6],
                "status": result[7],
            }
        return None

    def cancel_appointment(self, appointment_id):
        """Cancel an appointment."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT time_slot_id, status FROM appointments WHERE id = ?", (appointment_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return {"success": False, "message": "Appointment not found"}

        if result[1] == "cancelled":
            conn.close()
            return {"success": False, "message": "Appointment already cancelled"}

        time_slot_id = result[0]

        cursor.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,))
        cursor.execute("UPDATE time_slots SET is_available = 1 WHERE id = ?", (time_slot_id,))

        conn.commit()
        conn.close()

        return {"success": True, "message": "Appointment cancelled successfully"}

    def update_appointment(self, appointment_id, new_time_slot_id):
        """Update appointment to a new time slot."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT time_slot_id, status FROM appointments WHERE id = ?", (appointment_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return {"success": False, "message": "Appointment not found"}

        if result[1] == "cancelled":
            conn.close()
            return {"success": False, "message": "Cannot update cancelled appointment"}

        old_slot_id = result[0]

        cursor.execute("SELECT is_available, doctor_id FROM time_slots WHERE id = ?", (new_time_slot_id,))
        new_slot = cursor.fetchone()

        if not new_slot or not new_slot[0]:
            conn.close()
            return {"success": False, "message": "New time slot not available"}

        cursor.execute(
            "UPDATE appointments SET time_slot_id = ?, doctor_id = ? WHERE id = ?",
            (new_time_slot_id, new_slot[1], appointment_id)
        )

        cursor.execute("UPDATE time_slots SET is_available = 1 WHERE id = ?", (old_slot_id,))
        cursor.execute("UPDATE time_slots SET is_available = 0 WHERE id = ?", (new_time_slot_id,))

        conn.commit()
        conn.close()

        return {"success": True, "message": "Appointment updated successfully"}

    def list_specialties(self) -> List[str]:
        """Return all distinct specialties available in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT specialty FROM doctors ORDER BY specialty")
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results

    def list_doctors(self, specialty: str = None) -> List[Dict[str, Any]]:
        """Return all doctors, optionally filtered by specialty."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if specialty:
            cursor.execute(
                "SELECT id, name, specialty FROM doctors WHERE specialty LIKE ? ORDER BY specialty, name",
                (f"%{specialty}%",)
            )
        else:
            cursor.execute("SELECT id, name, specialty FROM doctors ORDER BY specialty, name")
        results = [{"id": r[0], "name": r[1], "specialty": r[2]} for r in cursor.fetchall()]
        conn.close()
        return results


# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
db = MedicalAppointmentDB()

# Quick smoke test
if __name__ == "__main__":
    print("=== Specialties in DB ===")
    for sp in db.list_specialties():
        print(f"  {sp}")

    print(f"\n=== Total doctors: {len(db.list_doctors())} ===")

    print("\n=== Sample availability (Cardiology) ===")
    for slot in db.check_availability(specialty="Cardiology")[:4]:
        print(f"  {slot['doctor']} | {slot['date']} {slot['time']}")

    print("\n=== Prompt snippet for Neurology ===")
    print(get_system_prompt("neurology")[:200], "...")