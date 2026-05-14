import pandas as pd
import logging
from datetime import datetime

# Set up logging - I added this after my supervisor mentioned audit trails
# during the feedback session, hadn't thought about it before.
logging.basicConfig(
    filename="triage_rpa.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_FILE  = "patient_arrivals.xlsx"
OUTPUT_FILE = "triage_output.xlsx"

# Triage level names from MTS (Manchester Triage System)
# Source: Mackway-Jones et al. 2014, also cross-checked with the 2023 NHS guidance
LEVEL_NAMES = {
    1: "Immediate (Red)",
    2: "Very Urgent (Orange)",
    3: "Urgent (Yellow)",
    4: "Standard (Green)",
    5: "Non-Urgent (Blue)"
}

# --- vital sign thresholds ---
# I got most of these from the MTS handbook but a few I double-checked
# with this paper: Farrohknia et al. (2011) - Emergency Department Triage Scales
# the SpO2 cutoffs were the trickiest to decide on

HR_LOW   = 50    # below this = bradycardia risk
HR_HIGH  = 130   # above = tachycardia
HR_MID   = 110   # mildly elevated

BP_CRIT  = 90    # hypotension cutoff
BP_LOW   = 100

SPO2_BAD = 90    # below this is basically an emergency
SPO2_OK  = 94    # borderline - still needs attention

TEMP_HIGH  = 39.5
TEMP_MID   = 38.5

GCS_BAD  = 8     # below 8 = can't protect airway, learned this in the lecture

AGE_ELDERLY = 65


def load_data(filepath):
    # Try to read the file, if not there use the fake data I made for testing
    try:
        df = pd.read_excel(filepath)
        print(f"Loaded {len(df)} patient records.")
        logging.info(f"Loaded {len(df)} records from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        print("Using demo data instead.")
        logging.warning(f"{filepath} not found - using demo data")
        return make_demo_data()
    except Exception as err:
        print(f"Something went wrong loading the file: {err}")
        logging.error(f"Load error: {err}")
        return None


def make_demo_data():
    # I made these cases up but tried to make them realistic
    # P007 (unresponsive) and P003 (low SpO2) should come out as Level 1
    records = {
        "patient_id": ["P001","P002","P003","P004","P005",
                       "P006","P007","P008","P009","P010"],
        "age":        [67, 23, 45, 8, 55, 32, 78, 19, 60, 41],
        "chief_complaint": [
            "chest pain radiating to left arm",
            "minor cut on hand",
            "difficulty breathing",
            "high fever and rash",
            "severe sudden headache",
            "twisted ankle after fall",
            "found unresponsive at home",
            "stomach ache since morning",
            "facial droop and slurred speech",
            "lower back pain for two days"
        ],
        "heart_rate":   [110, 72, 125, 102, 88,  70, 45, 80, 95,  75],
        "systolic_bp":  [ 90,120, 100, 105,160, 118, 80,115, 85, 130],
        "spo2":         [ 94, 99,  88,  98, 97,  99, 91, 98, 93,  99],
        "temp_c":       [37.1,36.8,37.5,39.2,37.0,36.9,36.5,38.1,37.2,36.7],
        "gcs":          [ 15, 15,  14,  15, 15,  15,  8, 15, 13,  15],
        "arrival_time": pd.date_range(start="2024-01-15 08:00", periods=10, freq="12min")
    }
    return pd.DataFrame(records)


def score_patient(row):
    # Keyword matching at the bottom doesn't handle negations.
    # "no chest pain" still triggers chest pain - I know this is wrong
    # but proper NLP is what the AI agent is for, not this part.

    level = 5  # start at lowest urgency

    # GCS first - if they're barely conscious nothing else matters
    if row["gcs"] <= GCS_BAD:
        return 1

    # oxygen - below 90 is very bad
    if row["spo2"] < SPO2_BAD:
        level = min(level, 1)
    elif row["spo2"] < SPO2_OK:
        level = min(level, 2)

    # heart rate checks
    # I had this as two separate if statements at first which caused a bug
    # where someone could get bumped up twice - min() fixes that
    if row["heart_rate"] < HR_LOW or row["heart_rate"] > HR_HIGH:
        level = min(level, 2)
    elif row["heart_rate"] > HR_MID:
        level = min(level, 3)

    # blood pressure
    if row["systolic_bp"] < BP_CRIT:
        level = min(level, 1)
    elif row["systolic_bp"] < BP_LOW:
        level = min(level, 2)

    # temperature
    if row["temp_c"] >= TEMP_HIGH:
        level = min(level, 2)
    elif row["temp_c"] >= TEMP_MID:
        level = min(level, 3)

    # keyword check on free text complaint
    # as mentioned above this is the weak part of the RPA approach
    danger_words = [
        "chest pain", "stroke", "unresponsive", "difficulty breathing",
        "severe headache", "facial droop", "slurred speech"
    ]
    medium_words = ["high fever", "rash", "abdominal pain", "vomiting blood"]

    complaint = str(row["chief_complaint"]).lower()

    if any(w in complaint for w in danger_words):
        level = min(level, 2)
    elif any(w in complaint for w in medium_words):
        level = min(level, 3)

    # elderly patients - MTS recommends treating age as an escalation factor
    # I set the threshold at 65 based on the WHO definition of elderly
    if row["age"] >= AGE_ELDERLY and level > 2:
        level = level - 1

    return level


def alert_critical(df):
    # pull out level 1 patients and print a warning
    critical_pts = df[df["triage_level"] == 1]

    if not critical_pts.empty:
        print("")
        print("!" * 50)
        print(f"  WARNING: {len(critical_pts)} patient(s) need IMMEDIATE attention")
        print("!" * 50)
        for _, p in critical_pts.iterrows():
            print(f"  -> {p['patient_id']} | {p['chief_complaint']}")
            print(f"     SpO2: {p['spo2']}%  |  BP: {p['systolic_bp']} mmHg  |  GCS: {p['gcs']}")
            logging.warning(f"Critical: {p['patient_id']} - {p['chief_complaint']}")

    return critical_pts


def nurse_review(df):
    # Human in the Loop - nurse has to confirm before we save, otherwise we'd be making clinical decisions automatically
    critical_pts = df[df["triage_level"] == 1].copy()

    if critical_pts.empty:
        return df

    print("")
    print("--- NURSE REVIEW ---")
    print("The following patients were auto-assigned Level 1.")
    print("Please confirm or change each one before saving.\n")

    for idx, p in critical_pts.iterrows():
        print(f"Patient: {p['patient_id']}")
        print(f"  Complaint : {p['chief_complaint']}")
        print(f"  HR: {p['heart_rate']}  BP: {p['systolic_bp']}  SpO2: {p['spo2']}  GCS: {p['gcs']}")

        answer = input("  Keep Level 1? [Enter = yes, or type 2-5 to change]: ").strip()

        if answer == "":
            logging.info(f"Nurse confirmed Level 1 for {p['patient_id']}")
            print("  Kept as Level 1.\n")
        elif answer in ["2", "3", "4", "5"]:
            new_level = int(answer)
            df.at[idx, "triage_level"]   = new_level
            df.at[idx, "triage_label"]   = LEVEL_NAMES[new_level]
            df.at[idx, "nurse_override"] = True
            logging.info(f"Nurse changed {p['patient_id']} from Level 1 to Level {new_level}")
            print(f"  Changed to Level {answer}.\n")
        else:
            print("  Invalid input - keeping Level 1.\n")

    return df


def show_summary(df):
    print("")
    print("=" * 45)
    print("TRIAGE SUMMARY")
    print(datetime.now().strftime("%d %b %Y  %H:%M"))
    print("=" * 45)
    print(f"Patients processed: {len(df)}")
    print("")

    for lvl in sorted(df["triage_level"].unique()):
        n     = len(df[df["triage_level"] == lvl])
        name  = LEVEL_NAMES.get(lvl, "unknown")
        bar   = "█" * n
        print(f"  L{lvl} {name}: {bar} ({n})")

    high_risk = len(df[df["risk_flag"] == "HIGH"])
    print(f"\n  Total high-risk (L1-L2): {high_risk}")
    print("=" * 45)


def save_output(df, filepath):
    try:
        df.to_excel(filepath, index=False)
        print(f"\nSaved results to: {filepath}")
        logging.info(f"Output saved to {filepath}")
    except PermissionError:
        # Usually means the file is open in Excel
        print("Error: can't save - is the output file open somewhere?")
        logging.error(f"PermissionError saving {filepath}")
    except Exception as err:
        print(f"Save failed: {err}")
        logging.error(f"Save error: {err}")


def main():
    print("=" * 45)
    print("RPA TRIAGE SYSTEM  |  ED Decision Support")
    print("=" * 45)
    logging.info("--- Session started ---")

    df = load_data(INPUT_FILE)
    if df is None:
        print("Could not load data. Exiting.")
        return

    # Run the scoring function on every row
    df["triage_level"]   = df.apply(score_patient, axis=1)
    df["triage_label"]   = df["triage_level"].map(LEVEL_NAMES)
    df["nurse_override"] = False
    df["processed_at"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Added this after the review session - simple flag useful for reports
    df["risk_flag"] = df["triage_level"].apply(lambda x: "HIGH" if x <= 2 else "LOW")

    alert_critical(df)

    # Nurse checks critical patients before anything gets written to disk
    df = nurse_review(df)

    show_summary(df)
    save_output(df, OUTPUT_FILE)

    logging.info(f"Session ended - {len(df)} patients processed.")
    print("\nDone.")


if __name__ == "__main__":
    main()
