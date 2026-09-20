import pandas as pd
import numpy as np
import random
import os  #ADDED: to check if file exists
#unified metadata - using SingleTableMetadata for compatibility in this version
from datetime import date
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
#seeds just for reproducability sake
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#create a small "real" seed dataset (fake/manual prototype)

#synthetic data
SYNTHEA_DIR = 'data/10k_synthea_covid19_csv'

#loading patient data and calculating age
patients = pd.read_csv(f'{SYNTHEA_DIR}/patients.csv', usecols=['Id', 'BIRTHDATE', 'GENDER'])
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
today = pd.Timestamp(date.today())
#365.25 days in a year hence why we're dividing by 365.25
patients['age'] = ((today - patients['BIRTHDATE']).dt.days / 365.25).astype(int)
patients = patients.rename(columns={'Id': 'patient_id', 'GENDER': 'gender'})
patients = patients[['patient_id', 'age', 'gender']]

#loading observations and BP+HEARTRATE   /   they're LOINC codes which is a universal ID system for medical measurements
obs = pd.read_csv(f'{SYNTHEA_DIR}/observations.csv', usecols=['PATIENT', 'CODE', 'DESCRIPTION', 'VALUE', 'DATE'])

#8480-6 = systolic blood pressure (top number)
#8462-4 = diastolic blood pressure (bottom number)
#8867-4 = heart rate
bp_sys = obs[obs['CODE'] == '8480-6'][['PATIENT', 'VALUE', 'DATE']].rename(columns={'VALUE': 'blood_pressure_systolic'})
bp_dia = obs[obs['CODE'] == '8462-4'][['PATIENT', 'VALUE']].rename(columns={'VALUE': 'blood_pressure_diastolic'})
hr = obs[obs['CODE'] == '8867-4'][['PATIENT', 'VALUE']].rename(columns={'VALUE': 'heart_rate'})

#some records of patients have multiple readings. because of this I'm only going to use the most recent one
bp_sys = bp_sys.sort_values('DATE').groupby('PATIENT').last().reset_index()
bp_sys = bp_sys.rename(columns={'PATIENT': 'patient_id'})

bp_dia = bp_dia.groupby('PATIENT').last().reset_index().rename(columns={'PATIENT': 'patient_id'})
hr = hr.groupby('PATIENT').last().reset_index().rename(columns={'PATIENT': 'patient_id'})

#join all three measurements together into one table per patient
vitals = bp_sys[['patient_id', 'blood_pressure_systolic', 'DATE']].merge(bp_dia, on='patient_id').merge(hr, on='patient_id')
vitals = vitals.rename(columns={'DATE': 'visit_date'})

#confirming that numbers are being stored as numbers as opposed to text and dropping any rows where a value is missing
vitals['blood_pressure_systolic'] = pd.to_numeric(vitals['blood_pressure_systolic'], errors='coerce')
vitals['blood_pressure_diastolic'] = pd.to_numeric(vitals['blood_pressure_diastolic'], errors='coerce')
vitals['heart_rate'] = pd.to_numeric(vitals['heart_rate'], errors='coerce')
vitals = vitals.dropna()

#conditions file lists every diagnosis a patient has ever had - i'm using SNOMED codes here ,which is another medical coding system, except this time for diagnoses
#44054006 = type 2 diabetes, 73211009 = diabetes mellitus (general)
conditions = pd.read_csv(f'{SYNTHEA_DIR}/conditions.csv', usecols=['PATIENT', 'CODE'])
diabetes_codes = ['44054006', '73211009', 44054006, 73211009]
diabetic_patients = conditions[conditions['CODE'].isin(diabetes_codes)]['PATIENT'].unique()

#joining data onto on table then adding a 1/0 column to flag whether each patient has diabetes
seed_data = patients.merge(vitals, on='patient_id')
seed_data['has_diabetes'] = seed_data['patient_id'].isin(diabetic_patients).astype(int)
seed_data['visit_date'] = pd.to_datetime(seed_data['visit_date'])

#for testing purposes i'm setting it up so that there's roughly an equal number of diabetic and non diabetic patients (seed data)
#main purpose for this is to prevent the model from ONLY learning how to predict "no diabetes" for everyone
diabetic = seed_data[seed_data['has_diabetes'] == 1].sample(min(150, seed_data['has_diabetes'].sum()), random_state=RANDOM_SEED)
nondiabetic = seed_data[seed_data['has_diabetes'] == 0].sample(150, random_state=RANDOM_SEED)
seed_data = pd.concat([diabetic, nondiabetic]).reset_index(drop=True)

print(f"Seed dataset: {len(seed_data)} patients ({seed_data['has_diabetes'].sum()} diabetic, {(seed_data['has_diabetes']==0).sum()} non-diabetic)")

#setting up SDV. first identifying what kind of data each column contains, then fit the model on the seed data so it learns the statistical patterns
os.makedirs('data', exist_ok=True)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=seed_data)

#override/update specific sdtypes where auto-detection might be wrong
metadata.update_column(column_name='patient_id', sdtype='id')
metadata.update_column(column_name='has_diabetes', sdtype='categorical')
metadata.update_column(column_name='visit_date', sdtype='datetime')
metadata.update_column(column_name='gender', sdtype='categorical')

#save to json ONLY if file does not already exist
metadata_path = 'data/ehr_metadata.json'
if not os.path.exists(metadata_path):
    metadata.save_to_json(metadata_path)
    print("metadata saved to data/ehr_metadata.json.")
else:
    print("metadata file already exists. Skipping save (to avoid overwrite warning).")

#using guassianCupola as synthesizer - learning patterns from synthea data
print("fitting GaussianCopulaSYnthesizer on Synthea seed data")
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(seed_data)

#generate 500 synthetic patients based on learned patterns
synthetic_data = synthesizer.sample(num_rows=2000)

#save it to CSV file
synthetic_data.to_csv('data/synthetic_ehr.csv', index=False)

#preview results
print(f"\nGenerated {len(synthetic_data)} synthetic records.")
print(f"Diabetes split: {synthetic_data['has_diabetes'].value_counts().to_dict()}")
print(synthetic_data.head())
