import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_HOSP_PATH = os.path.join(
    BASE_DIR,
    "..", "raw_mimic", "mimic-iv-clinical-database-demo-2.2", "hosp"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed_mimic_24h_labs_demographics.csv"
)


# Lab variables to keep
LAB_LABELS = [
    "Sodium",
    "Potassium",
    "Chloride",
    "Creatinine",
    "Urea Nitrogen",
    "Hematocrit",
    "Hemoglobin",
    "WBC",
    "Platelet Count",
    "Glucose",
]


# Load raw tables
def load_hosp_tables(raw_hosp_path=RAW_HOSP_PATH):
    print("Loading raw MIMIC hosp CSVs...")
    patients = pd.read_csv(os.path.join(raw_hosp_path, "patients.csv.gz"))
    admissions = pd.read_csv(os.path.join(raw_hosp_path, "admissions.csv.gz"))
    labevents = pd.read_csv(os.path.join(raw_hosp_path, "labevents.csv.gz"))
    labitems = pd.read_csv(os.path.join(raw_hosp_path, "d_labitems.csv.gz"))

    print("Loaded shapes:")
    print("  patients   :", patients.shape)
    print("  admissions :", admissions.shape)
    print("  labevents  :", labevents.shape)
    print("  labitems   :", labitems.shape)

    return patients, admissions, labevents, labitems


# Select lab events for chosen labels
def select_labs_24h(labevents, labitems, admissions, lab_labels):
    # Map labels in itemids
    labitems_sel = labitems[labitems["label"].isin(lab_labels)]
    itemids = labitems_sel["itemid"].unique().tolist()

    print(f"Selected {len(itemids)} itemids for {len(lab_labels)} lab labels.")

    # Keep only those lab events
    le = labevents[labevents["itemid"].isin(itemids)].copy()

    # Merge to get label names
    le = le.merge(
        labitems_sel[["itemid", "label"]],
        on="itemid",
        how="left"
    )

    # Merge with admissions to get admittime
    # (inner join on subject_id, hadm_id)
    le = le.merge(
        admissions[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"],
        how="inner"
    )


    le["charttime"] = pd.to_datetime(le["charttime"])
    le["admittime"] = pd.to_datetime(le["admittime"])

    # only first 24h after admission
    le_24h = le[
        (le["charttime"] >= le["admittime"]) &
        (le["charttime"] <= le["admittime"] + pd.Timedelta(hours=24))
    ].copy()

    print("Lab events in first 24h:", le_24h.shape)

    return le_24h


# Aggregate labs per admission
def aggregate_labs(le_24h):
    # Median per (subject_id, hadm_id, label)
    agg = (
        le_24h
        .groupby(["subject_id", "hadm_id", "label"])["valuenum"]
        .median()
        .reset_index()
    )

    # Pivot to wide format
    labs_pivot = agg.pivot_table(
        index=["subject_id", "hadm_id"],
        columns="label",
        values="valuenum"
    ).reset_index()


    labs_pivot.columns.name = None

    print("Aggregated labs pivot shape:", labs_pivot.shape)
    return labs_pivot


# Merge with admissions and patients

def merge_with_context(labs_pivot, admissions, patients):
    df = labs_pivot.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="left"
    )

    df = df.merge(
        patients,
        on="subject_id",
        how="left"
    )

    # Convert times
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["edregtime"] = pd.to_datetime(df["edregtime"])
    df["edouttime"] = pd.to_datetime(df["edouttime"])

    # Length of stay in hours
    df["length_of_stay_hours"] = (
        (df["dischtime"] - df["admittime"])
        .dt.total_seconds() / 3600.0
    )

    # ED wait time in hours
    df["ed_wait_time_hours"] = (
        (df["edouttime"] - df["edregtime"])
        .dt.total_seconds() / 3600.0
    )

    return df


# Select final variables
def select_final_variables(df):

    lab_cols = [col for col in LAB_LABELS if col in df.columns]


    demo_cols = [
        "gender",
        "anchor_age",
        "anchor_year_group",
    ]


    adm_cols = [
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
        "length_of_stay_hours",
        "ed_wait_time_hours",
    ]

    keep_cols = ["subject_id", "hadm_id"] + lab_cols + demo_cols + adm_cols


    keep_cols = [c for c in keep_cols if c in df.columns]

    df_final = df[keep_cols].copy()

    print("\nFinal variable set:")
    print(df_final.columns.tolist())
    print("Final shape:", df_final.shape)


    print("\nMissingness (fraction NaN per column):")
    print(df_final.isna().mean())

    return df_final


def save_dataset(df, path=OUTPUT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved processed dataset to: {path}")


def main():
    patients, admissions, labevents, labitems = load_hosp_tables()

    le_24h = select_labs_24h(
        labevents=labevents,
        labitems=labitems,
        admissions=admissions,
        lab_labels=LAB_LABELS,
    )

    labs_pivot = aggregate_labs(le_24h)

    df_full = merge_with_context(
        labs_pivot=labs_pivot,
        admissions=admissions,
        patients=patients,
    )

    df_final = select_final_variables(df_full)

    save_dataset(df_final)


if __name__ == "__main__":
    main()
