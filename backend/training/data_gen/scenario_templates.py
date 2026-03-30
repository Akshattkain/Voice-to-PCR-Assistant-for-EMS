"""EMS scenario templates for synthetic data generation.

Defines 14 scenario types with parameter distributions for generating
realistic (transcript, PCR JSON) pairs.
"""

from pydantic import BaseModel


class ScenarioTemplate(BaseModel):
    """Template for generating synthetic EMS scenarios."""

    type: str
    description: str
    age_range: tuple[int, int]
    sex_distribution: dict[str, float]
    common_chief_complaints: list[str]
    common_impressions: list[str]
    typical_vitals: dict[str, tuple[int, int]]  # field -> (low, high)
    common_medications_given: list[str]
    common_procedures: list[str]
    common_allergies: list[str]
    common_pmh: list[str]
    common_signs_symptoms: list[str]


SCENARIO_TEMPLATES: dict[str, ScenarioTemplate] = {
    "cardiac": ScenarioTemplate(
        type="cardiac",
        description="Cardiac emergencies: chest pain, STEMI, arrhythmia, heart failure",
        age_range=(40, 90),
        sex_distribution={"male": 0.6, "female": 0.4},
        common_chief_complaints=[
            "Chest Pain/Discomfort", "Palpitations", "Shortness of Breath",
            "Syncope", "Cardiac Arrest",
        ],
        common_impressions=[
            "Angina pectoris, unspecified", "Acute myocardial infarction",
            "Atrial fibrillation", "Congestive heart failure",
            "Cardiac arrest", "Supraventricular tachycardia",
        ],
        typical_vitals={
            "bp_systolic": (90, 200), "bp_diastolic": (50, 120),
            "heart_rate": (40, 180), "respiratory_rate": (12, 28),
            "spo2": (85, 99), "gcs_total": (3, 15),
        },
        common_medications_given=[
            "Aspirin", "Nitroglycerin", "Amiodarone", "Epinephrine",
            "Heparin", "Morphine", "Oxygen",
        ],
        common_procedures=[
            "12 lead electrocardiogram", "Catheterization of vein",
            "Cardiac monitoring", "Defibrillation", "CPR",
        ],
        common_allergies=["penicillin", "sulfa", "NKDA", "iodine", "aspirin"],
        common_pmh=["CAD", "hypertension", "diabetes", "CHF", "atrial fibrillation", "CABG"],
        common_signs_symptoms=[
            "Chest pain, unspecified", "Diaphoresis", "Dyspnea",
            "Nausea", "Radiating pain to left arm", "Pallor",
        ],
    ),
    "respiratory": ScenarioTemplate(
        type="respiratory",
        description="Respiratory emergencies: asthma, COPD, pneumonia, PE",
        age_range=(20, 85),
        sex_distribution={"male": 0.5, "female": 0.5},
        common_chief_complaints=[
            "Shortness of Breath", "Difficulty Breathing", "Wheezing", "Cough",
        ],
        common_impressions=[
            "Asthma exacerbation", "COPD exacerbation", "Pneumonia",
            "Pulmonary embolism", "Respiratory failure",
        ],
        typical_vitals={
            "bp_systolic": (100, 180), "bp_diastolic": (60, 100),
            "heart_rate": (80, 140), "respiratory_rate": (20, 40),
            "spo2": (70, 96), "gcs_total": (12, 15),
        },
        common_medications_given=[
            "Albuterol", "Ipratropium", "Methylprednisolone", "Oxygen",
            "Epinephrine", "Magnesium sulfate",
        ],
        common_procedures=[
            "Nebulizer treatment", "CPAP", "Endotracheal intubation",
            "Pulse oximetry", "Catheterization of vein",
        ],
        common_allergies=["NKDA", "penicillin", "NSAIDs"],
        common_pmh=["COPD", "asthma", "CHF", "smoking history", "lung cancer"],
        common_signs_symptoms=[
            "Wheezing", "Dyspnea", "Tachypnea", "Accessory muscle use",
            "Cyanosis", "Tripod positioning",
        ],
    ),
    "trauma_blunt": ScenarioTemplate(
        type="trauma_blunt",
        description="Blunt trauma: MVC, falls, assaults",
        age_range=(16, 80),
        sex_distribution={"male": 0.65, "female": 0.35},
        common_chief_complaints=[
            "Traumatic Injury", "Fall", "Motor Vehicle Collision",
            "Pain after injury", "Head Injury",
        ],
        common_impressions=[
            "Closed fracture", "Concussion", "Contusion",
            "Spinal cord injury", "Internal hemorrhage",
        ],
        typical_vitals={
            "bp_systolic": (70, 160), "bp_diastolic": (40, 100),
            "heart_rate": (60, 140), "respiratory_rate": (12, 30),
            "spo2": (88, 99), "gcs_total": (3, 15),
        },
        common_medications_given=[
            "Normal saline", "Fentanyl", "Morphine", "Ketamine",
            "Tranexamic acid", "Oxygen",
        ],
        common_procedures=[
            "Spinal immobilization", "Splinting", "Catheterization of vein",
            "Wound care", "Cervical collar application",
        ],
        common_allergies=["NKDA", "codeine", "penicillin", "latex"],
        common_pmh=["none", "hypertension", "diabetes", "anticoagulation therapy"],
        common_signs_symptoms=[
            "Pain at injury site", "Deformity", "Swelling", "Ecchymosis",
            "Limited range of motion", "Altered mental status",
        ],
    ),
    "neurological": ScenarioTemplate(
        type="neurological",
        description="Neurological emergencies: stroke, seizure, syncope",
        age_range=(30, 90),
        sex_distribution={"male": 0.5, "female": 0.5},
        common_chief_complaints=[
            "Stroke/CVA", "Seizure", "Altered Mental Status",
            "Syncope", "Headache", "Weakness",
        ],
        common_impressions=[
            "Cerebrovascular accident", "Seizure disorder",
            "Transient ischemic attack", "Altered mental status",
        ],
        typical_vitals={
            "bp_systolic": (100, 220), "bp_diastolic": (60, 130),
            "heart_rate": (50, 120), "respiratory_rate": (10, 24),
            "spo2": (90, 99), "gcs_total": (3, 15),
        },
        common_medications_given=[
            "Oxygen", "Midazolam", "Lorazepam", "Normal saline", "Dextrose",
        ],
        common_procedures=[
            "Blood glucose monitoring", "Stroke assessment (Cincinnati/LAMS)",
            "Catheterization of vein", "Cardiac monitoring",
        ],
        common_allergies=["NKDA", "penicillin", "contrast dye"],
        common_pmh=["hypertension", "atrial fibrillation", "prior stroke", "seizure disorder", "diabetes"],
        common_signs_symptoms=[
            "Facial droop", "Arm drift", "Speech abnormality",
            "Unilateral weakness", "Confusion", "Headache",
        ],
    ),
    "diabetic": ScenarioTemplate(
        type="diabetic",
        description="Diabetic emergencies: hypoglycemia, DKA",
        age_range=(20, 80),
        sex_distribution={"male": 0.5, "female": 0.5},
        common_chief_complaints=[
            "Diabetic Problem", "Altered Mental Status", "Weakness",
            "Nausea/Vomiting", "Syncope",
        ],
        common_impressions=[
            "Hypoglycemia", "Diabetic ketoacidosis",
            "Hyperglycemia", "Altered mental status due to diabetes",
        ],
        typical_vitals={
            "bp_systolic": (90, 170), "bp_diastolic": (50, 100),
            "heart_rate": (70, 130), "respiratory_rate": (14, 32),
            "spo2": (92, 99), "gcs_total": (6, 15),
        },
        common_medications_given=[
            "Dextrose 50%", "Glucagon", "Normal saline", "Insulin", "Oxygen",
        ],
        common_procedures=[
            "Blood glucose monitoring", "Catheterization of vein",
            "Cardiac monitoring",
        ],
        common_allergies=["NKDA", "sulfa", "penicillin"],
        common_pmh=["diabetes type 1", "diabetes type 2", "hypertension", "renal disease"],
        common_signs_symptoms=[
            "Altered mental status", "Diaphoresis", "Tremors",
            "Kussmaul breathing", "Fruity breath odor", "Nausea",
        ],
    ),
    "allergic_reaction": ScenarioTemplate(
        type="allergic_reaction",
        description="Allergic reactions and anaphylaxis",
        age_range=(5, 70),
        sex_distribution={"male": 0.45, "female": 0.55},
        common_chief_complaints=[
            "Allergic Reaction", "Anaphylaxis", "Difficulty Breathing",
            "Swelling", "Hives",
        ],
        common_impressions=[
            "Anaphylaxis", "Allergic reaction", "Angioedema", "Urticaria",
        ],
        typical_vitals={
            "bp_systolic": (60, 150), "bp_diastolic": (30, 90),
            "heart_rate": (80, 160), "respiratory_rate": (16, 36),
            "spo2": (80, 99), "gcs_total": (10, 15),
        },
        common_medications_given=[
            "Epinephrine", "Diphenhydramine", "Methylprednisolone",
            "Albuterol", "Normal saline",
        ],
        common_procedures=[
            "Catheterization of vein", "Cardiac monitoring",
            "EpiPen administration",
        ],
        common_allergies=["peanuts", "shellfish", "bee stings", "penicillin", "latex"],
        common_pmh=["prior anaphylaxis", "asthma", "eczema", "food allergies"],
        common_signs_symptoms=[
            "Urticaria", "Angioedema", "Stridor", "Wheezing",
            "Hypotension", "Tachycardia", "Pruritus",
        ],
    ),
    "overdose": ScenarioTemplate(
        type="overdose",
        description="Drug overdose and poisoning",
        age_range=(16, 65),
        sex_distribution={"male": 0.6, "female": 0.4},
        common_chief_complaints=[
            "Overdose/Poisoning", "Altered Mental Status",
            "Unresponsive", "Respiratory Depression",
        ],
        common_impressions=[
            "Opioid overdose", "Drug overdose, unspecified",
            "Alcohol intoxication", "Benzodiazepine overdose",
        ],
        typical_vitals={
            "bp_systolic": (70, 150), "bp_diastolic": (40, 90),
            "heart_rate": (40, 130), "respiratory_rate": (4, 20),
            "spo2": (60, 98), "gcs_total": (3, 14),
        },
        common_medications_given=[
            "Naloxone", "Oxygen", "Normal saline", "Activated charcoal", "Flumazenil",
        ],
        common_procedures=[
            "Catheterization of vein", "Bag-valve-mask ventilation",
            "Nasal airway insertion", "Cardiac monitoring",
        ],
        common_allergies=["NKDA", "unknown"],
        common_pmh=["substance abuse", "depression", "prior overdose", "hepatitis C"],
        common_signs_symptoms=[
            "Altered mental status", "Respiratory depression",
            "Miosis", "Bradycardia", "Cyanosis",
        ],
    ),
    "pediatric": ScenarioTemplate(
        type="pediatric",
        description="Pediatric emergencies",
        age_range=(0, 17),
        sex_distribution={"male": 0.52, "female": 0.48},
        common_chief_complaints=[
            "Fever", "Difficulty Breathing", "Seizure", "Injury",
            "Allergic Reaction", "Abdominal Pain",
        ],
        common_impressions=[
            "Febrile seizure", "Croup", "Asthma exacerbation",
            "Dehydration", "Fracture",
        ],
        typical_vitals={
            "bp_systolic": (70, 120), "bp_diastolic": (40, 80),
            "heart_rate": (80, 180), "respiratory_rate": (16, 50),
            "spo2": (88, 99), "gcs_total": (8, 15),
        },
        common_medications_given=[
            "Albuterol", "Epinephrine", "Acetaminophen", "Normal saline", "Oxygen",
        ],
        common_procedures=[
            "Pediatric assessment", "Nebulizer treatment",
            "Catheterization of vein", "Temperature assessment",
        ],
        common_allergies=["NKDA", "penicillin", "eggs"],
        common_pmh=["asthma", "none", "prematurity", "seizure disorder"],
        common_signs_symptoms=[
            "Fever", "Crying", "Lethargy", "Wheezing", "Rash", "Vomiting",
        ],
    ),
    "abdominal": ScenarioTemplate(
        type="abdominal",
        description="Abdominal emergencies: acute abdomen, GI bleed",
        age_range=(20, 85),
        sex_distribution={"male": 0.5, "female": 0.5},
        common_chief_complaints=[
            "Abdominal Pain", "Nausea/Vomiting", "GI Bleeding",
            "Diarrhea",
        ],
        common_impressions=[
            "Acute abdomen", "GI hemorrhage", "Appendicitis",
            "Bowel obstruction", "Cholecystitis",
        ],
        typical_vitals={
            "bp_systolic": (80, 170), "bp_diastolic": (50, 100),
            "heart_rate": (60, 130), "respiratory_rate": (14, 24),
            "spo2": (94, 99), "gcs_total": (13, 15),
        },
        common_medications_given=[
            "Ondansetron", "Normal saline", "Fentanyl", "Morphine", "Oxygen",
        ],
        common_procedures=[
            "Catheterization of vein", "Cardiac monitoring", "Pain assessment",
        ],
        common_allergies=["NKDA", "penicillin", "NSAIDs", "codeine"],
        common_pmh=["GERD", "prior surgery", "GI bleed history", "cirrhosis", "ulcer disease"],
        common_signs_symptoms=[
            "Abdominal tenderness", "Guarding", "Nausea", "Vomiting",
            "Melena", "Hematemesis",
        ],
    ),
    "cardiac_arrest": ScenarioTemplate(
        type="cardiac_arrest",
        description="Cardiac arrest: witnessed/unwitnessed",
        age_range=(30, 90),
        sex_distribution={"male": 0.65, "female": 0.35},
        common_chief_complaints=["Cardiac Arrest", "Unresponsive", "Not Breathing"],
        common_impressions=[
            "Cardiac arrest - ventricular fibrillation",
            "Cardiac arrest - PEA",
            "Cardiac arrest - asystole",
        ],
        typical_vitals={
            "bp_systolic": (0, 0), "bp_diastolic": (0, 0),
            "heart_rate": (0, 0), "respiratory_rate": (0, 0),
            "spo2": (0, 60), "gcs_total": (3, 3),
        },
        common_medications_given=[
            "Epinephrine", "Amiodarone", "Normal saline", "Sodium bicarbonate",
        ],
        common_procedures=[
            "CPR", "Defibrillation", "Endotracheal intubation",
            "Catheterization of vein", "Cardiac monitoring", "IO access",
        ],
        common_allergies=["NKDA", "unknown"],
        common_pmh=["CAD", "CHF", "hypertension", "unknown"],
        common_signs_symptoms=[
            "Unresponsive", "Pulseless", "Apneic", "Cyanosis",
        ],
    ),
}
