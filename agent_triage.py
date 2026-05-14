import json
import os
import time
import logging
from datetime import datetime

logging.basicConfig(
    filename="triage_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Run: pip install openai")
    exit(1)

API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
client  = OpenAI(api_key=API_KEY)

MAX_RETRIES    = 2
FALLBACK_LEVEL = 2  # if API completely fails, default to Very Urgent - safer than guessing lower

# --- system prompt ---
# took me a while to get this right. first version just said "you are a triage assistant"
# and the model kept giving generic health advice instead of an actual MTS level.
# adding the strict JSON requirement and listing out all 5 levels explicitly fixed it.
# I also tried gpt-4o first but switched to mini because it was getting expensive
# for testing and the outputs were similar quality for this task.
SYSTEM_PROMPT = """You are an AI triage decision support agent in a hospital emergency department.
Your role is to assist triage nurses by analysing patient presentations and suggesting
an initial acuity level based on the Manchester Triage System (MTS).

MTS levels:
  1 = Immediate (Red)      - treat within 0 min
  2 = Very Urgent (Orange) - treat within 10 min
  3 = Urgent (Yellow)      - treat within 60 min
  4 = Standard (Green)     - treat within 120 min
  5 = Non-Urgent (Blue)    - treat within 240 min

Rules:
- Identify red flag symptoms clearly
- Your output is a support tool only - nurse has final say
- Be concise, not verbose

Respond ONLY with a valid JSON object, no extra text:
{
  "suggested_triage_level": <1-5>,
  "triage_label": <string>,
  "confidence": <"high"|"medium"|"low">,
  "red_flags_identified": [<strings>],
  "clinical_reasoning": <string>,
  "recommended_immediate_actions": [<strings>],
  "disclaimer": "AI suggestion only. Nurse decision overrides."
}"""


def format_patient_prompt(patient):
    # just puts the patient dict into a readable block for the model
    # I tried passing raw JSON once but the model did better with plain text
    return f"""Assess this patient:

ID: {patient.get('patient_id')}
Age: {patient.get('age')} years
Complaint: {patient.get('chief_complaint')}

Vitals:
  HR:   {patient.get('heart_rate')} bpm
  BP:   {patient.get('systolic_bp')} mmHg
  SpO2: {patient.get('spo2')}%
  Temp: {patient.get('temp_c')} C
  GCS:  {patient.get('gcs')}/15

Notes: {patient.get('notes', 'none')}"""


def extract_json(raw_text):
    # the model sometimes wraps the JSON in markdown code blocks
    # this happened constantly in early testing and kept breaking everything
    # so I added this cleanup step - feels hacky but it works
    text = raw_text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.split("```")[0]
    return json.loads(text.strip())


def ask_agent(patient):
    pid = patient.get("patient_id", "?")
    print(f"\n[agent] processing {pid}...")
    logging.info(f"Sending {pid} to model")

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": format_patient_prompt(patient)}
                ],
                temperature=0.2,  # low temp = more consistent, less creative - better for clinical stuff
                max_tokens=500
            )
            raw = resp.choices[0].message.content
            result = extract_json(raw)
            logging.info(f"{pid} -> Level {result.get('suggested_triage_level')}, conf={result.get('confidence')}")
            return result

        except json.JSONDecodeError:
            # this happened a lot before I added the extract_json cleanup
            print(f"  attempt {attempt}: JSON parse failed, retrying...")
            logging.warning(f"{pid} attempt {attempt}: JSON error")
            time.sleep(1)

        except Exception as err:
            print(f"  attempt {attempt}: API error - {err}")
            logging.warning(f"{pid} attempt {attempt}: {err}")
            time.sleep(2)

    # if we get here all retries failed
    # defaulting to Level 2 because in an ED it's better to over-triage
    # than miss something serious - saw this principle mentioned in
    # FitzGerald et al. (2010) emergency triage paper
    print(f"  all retries failed for {pid}, assigning fallback Level {FALLBACK_LEVEL}")
    logging.error(f"{pid}: retries exhausted, fallback to Level {FALLBACK_LEVEL}")

    return {
        "suggested_triage_level": FALLBACK_LEVEL,
        "triage_label": "Very Urgent (Orange) [FALLBACK - assessment failed]",
        "confidence": "low",
        "red_flags_identified": ["Automated assessment unavailable - manual review required"],
        "clinical_reasoning": "API assessment failed. Safe fallback applied.",
        "recommended_immediate_actions": ["Manual triage required immediately"],
        "disclaimer": "AI suggestion only. Nurse decision overrides.",
        "is_fallback": True
    }


def show_result(pid, result):
    level_icons = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}

    print("\n" + "=" * 52)
    print(f"Patient: {pid}")
    print("=" * 52)

    lvl  = result.get("suggested_triage_level", "?")
    conf = result.get("confidence", "?").upper()
    icon = level_icons.get(lvl, "⚪")

    print(f"{icon}  Level {lvl} - {result.get('triage_label', '')}  [{conf} confidence]")

    if conf == "LOW":
        print("  ⚠️  low confidence - nurse should review carefully")

    flags = result.get("red_flags_identified", [])
    if flags:
        print("\nRed flags:")
        for f in flags:
            print(f"  - {f}")

    print(f"\nReasoning: {result.get('clinical_reasoning', '')}")

    actions = result.get("recommended_immediate_actions", [])
    if actions:
        print("\nSuggested actions:")
        for a in actions:
            print(f"  -> {a}")

    print(f"\n{result.get('disclaimer', '')}")
    print("=" * 52)


def nurse_check(pid, result):
    """
    After the agent gives a suggestion, the nurse can accept or override it.
    If confidence is low OR it's a fallback result, the nurse has to
    actively type something - they can't just press Enter and skip it, can't skip if confidence is low. Too risky.
    """
    suggested   = result.get("suggested_triage_level")
    conf        = result.get("confidence", "high")
    is_fallback = result.get("is_fallback", False)
    must_review = (conf == "low" or is_fallback)

    if must_review:
        print(f"\n[mandatory review] Low confidence or failed assessment for {pid}.")
        print("  Cannot auto-accept. Please enter a triage level.")
        answer = input("  Level (1-5): ").strip()
    else:
        print(f"\n[nurse review] Agent says Level {suggested} for {pid}.")
        answer = input("  Accept? [Enter = yes, or type 1-5 to change]: ").strip()

    if answer == "" and not must_review:
        result["final_level"]    = suggested
        result["nurse_override"] = False
        logging.info(f"{pid}: nurse accepted Level {suggested}")
        print("  Accepted.")

    elif answer in ["1", "2", "3", "4", "5"]:
        result["final_level"]    = int(answer)
        result["nurse_override"] = True
        logging.info(f"{pid}: nurse set Level {answer} (was {suggested})")
        print(f"  Set to Level {answer}.")

    else:
        # invalid input or mandatory review with no valid answer
        result["final_level"]    = FALLBACK_LEVEL
        result["nurse_override"] = False
        print(f"  No valid input - keeping Level {FALLBACK_LEVEL}.")
        logging.warning(f"{pid}: invalid nurse input, kept Level {FALLBACK_LEVEL}")

    result["reviewed_at"] = datetime.now().isoformat()
    return result


def save_results(results, filepath="agent_assessments.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Results saved to {filepath}")
    print(f"\nSaved to {filepath}")


# test patients - same IDs as the RPA script so outputs can be compared
TEST_PATIENTS = [
    {
        "patient_id": "P001",
        "age":        67,
        "chief_complaint": "chest pain radiating to left arm, started about 20 minutes ago",
        "heart_rate": 110,
        "systolic_bp": 90,
        "spo2":       94,
        "temp_c":     37.1,
        "gcs":        15,
        "notes": "Pale and sweaty. Has hypertension and diabetes. Says it feels like pressure."
    },
    {
        "patient_id": "P002",
        "age":        23,
        "chief_complaint": "cut on right hand from kitchen knife",
        "heart_rate": 72,
        "systolic_bp": 120,
        "spo2":       99,
        "temp_c":     36.8,
        "gcs":        15,
        "notes": "About 2cm cut. Bleeding stopped with pressure. Patient is fine, just nervous."
    },
    {
        "patient_id": "P003",
        "age":        78,
        "chief_complaint": "sudden confusion and slurred speech",
        "heart_rate": 95,
        "systolic_bp": 185,
        "spo2":       93,
        "temp_c":     37.2,
        "gcs":        12,
        "notes": "Family brought her in. Started about 45 min ago. No stroke history they know of."
    }
]


def main():
    print("=" * 52)
    print("AI TRIAGE AGENT  |  ED Decision Support  |  v1.0")
    print("=" * 52)
    logging.info("--- Agent session started ---")

    if API_KEY == "your-api-key-here":
        print("\nNo API key set - running mock demo.\n")
        print("Set key with:  export OPENAI_API_KEY='sk-...'")

        # show what the output looks like without needing a real API call
        mock = {
            "suggested_triage_level": 1,
            "triage_label": "Immediate (Red)",
            "confidence": "high",
            "red_flags_identified": [
                "BP 90 mmHg - hypotensive",
                "Chest pain radiating to arm - ACS presentation",
                "Diaphoresis"
            ],
            "clinical_reasoning": (
                "67-year-old with classic ACS features: crushing chest pain with "
                "left arm radiation, hypotension, and diaphoresis. Immediate intervention required."
            ),
            "recommended_immediate_actions": [
                "12-lead ECG immediately",
                "IV access and draw troponin/FBC",
                "Cardiac monitor",
                "Alert senior clinician now"
            ],
            "disclaimer": "AI suggestion only. Nurse decision overrides."
        }
        show_result("P001 (DEMO)", mock)
        nurse_check("P001 (DEMO)", mock)
        return

    all_results = []

    for patient in TEST_PATIENTS:
        pid    = patient["patient_id"]
        result = ask_agent(patient)
        show_result(pid, result)
        result = nurse_check(pid, result)
        all_results.append({
            "patient_id":  pid,
            "assessed_at": datetime.now().isoformat(),
            "result":      result
        })

    save_results(all_results)
    logging.info(f"Session ended - {len(all_results)} patients processed.")
    print(f"\nDone. {len(all_results)} patients assessed.")


if __name__ == "__main__":
    main()
