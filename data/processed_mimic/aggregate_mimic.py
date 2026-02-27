import pandas as pd
import os


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_PATH = "data/raw_mimic/mimic-iv-clinical-database-demo-2.2/hosp"  
PROCESSED_PATH = os.path.join(BASE_DIR)
OUTPUT_FILE = "processed_admissions_selected_labs.csv"


# Load raw MIMIC tables
def load_mimic_tables(raw_path=RAW_PATH):
    print("Loading raw MIMIC CSVs...")
    patients = pd.read_csv(os.path.join(raw_path, "patients.csv.gz"))
    admissions = pd.read_csv(os.path.join(raw_path, "admissions.csv.gz"))
    labevents = pd.read_csv(os.path.join(raw_path, "labevents.csv.gz"))
    labitems = pd.read_csv(os.path.join(raw_path, "d_labitems.csv.gz"))
    print(f"Patients: {patients.shape}, Admissions: {admissions.shape}, Lab events: {labevents.shape}, Lab items: {labitems.shape}")
    return patients, admissions, labevents, labitems


# Filter labevents for selected labs
def select_labs(labevents, labitems, selected_labs):
    """Filter labevents to only include selected lab tests"""
    itemids = labitems[labitems['label'].isin(selected_labs)]['itemid'].tolist()
    filtered = labevents[labevents['itemid'].isin(itemids)]
    filtered = filtered.merge(labitems[['itemid','label']], on='itemid', how='left')
    return filtered


# Aggregate per admission
def aggregate_labs(filtered_labevents):
    """Aggregate lab measurements per admission (mean of repeated measurements)"""
    agg = filtered_labevents.groupby(['subject_id','hadm_id','label'])['valuenum'].mean().reset_index()
    pivot = agg.pivot_table(index=['subject_id','hadm_id'], columns='label', values='valuenum').reset_index()
    return pivot


# Merge with admissions and patients
def merge_with_patients_admissions(pivot_labs, patients, admissions):
    """Merge aggregated labs with admissions and patient info"""
    df = pivot_labs.merge(admissions, on=['subject_id','hadm_id'], how='left')
    df = df.merge(patients, on='subject_id', how='left')
    return df


# Save cleaned dataset
def save_dataset(df, path=None):
    """Save processed dataset to CSV"""
    if path is None:
        path = os.path.join(PROCESSED_PATH, OUTPUT_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed dataset to: {path}")


# Main pipeline
def main(selected_labs=None):
    if selected_labs is None:
        selected_labs = ['Potassium', 'Sodium', 'Creatinine', 'Chloride', 'Urea Nitrogen', 'Hematocrit']

    patients, admissions, labevents, labitems = load_mimic_tables()
    filtered_labs = select_labs(labevents, labitems, selected_labs)
    admission_labs = aggregate_labs(filtered_labs)
    full_data = merge_with_patients_admissions(admission_labs, patients, admissions)
    
    # Print missing values summary
    print("\nMissing values per column:")
    print(full_data.isnull().sum())
    
    save_dataset(full_data)


if __name__ == "__main__":
    main()
