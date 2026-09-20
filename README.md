Privacy Enhancing Technologies for Electronic Health Records
Joey Hall (2205024)


Brief Overview:
A privacy-preserving EHR prototype that combines:
- A synthetic patient data generation (SDV / Gaussian Copula)
- Federated Learning simulation across 10 simulated hospital silos (FedAvg)
- Secure Flask web interface with authentication and audit logging


No real patient data is used at any point in this project.



Files:
generate_synthetic_ehr.py   - generates synthetic EHR data from Synthea Source
ehrsoft.py  - main flask web app
fl_tain.py  - federated learning simulation (FedAvg)
templates/login.html - login page template
templates/patients.html - patient records page template



This environment was tested on the following:
OS: Ubuntu 24.04 LTS
Virtualisation: VMware Workstation Pro 17
VM RAM: Minimum 8GB (16GB used in development)
VM Storage: Minimum 20GB free
Python Version: 3.12


SETUP INSTRUCTIONS



Creating project folder and move your files into
mkdir -p ~/ehr-project/data
mkdir -p ~/ehr-project/templates
mkdir -p ~/ehr-project/logs
mkdir -p ~/ehr-project/models
cd ~/ehr-project


Download synthea source data:

cd ~/ehr-project/data
wget https://synthetichealth.github.io/synthea-sample-data/downloads/10k_synthea_covid19_csv.zip
unzip 10k_synthea_covid19_csv.zip
cd ~/ehr-project



Create the python virtual env.:

python3 -m venv env



Activate the virtual environment:

source env/bin/activate



Install the required python libraries:

pip install flask sdv==1.17.2 torch pandas numpy scikit-learn werkzeug




RUNNING THE PROJECT

Make sure you're in ~/ehr-project with the virtual env. active before running any of the following commands.

1: Generate synthetic patient data

python generate_synthetic_ehr.py

2: Start the web application

python ehrsoft.py

3: Open a browser and go to

http://127.0.0.1:5000


LOGIN CREDENTIALS

Username: doctor
Password: password123


Username: admin
Password: adminpass


The training button can be found at the bottom of the page once logged in. Depending on your system specs, the time it'll take to complete may fluctuate. In this particular instance, the default settings use 10 hospital silos and 2000 synthetic patients. IF YOUR SYSTEM CRASHES OR RUNS OUT OF MEMORY you can reduce these settings as follows:

In generate_synthetic_ehr.py find the following line: 

synthetic_data = synthesizer.sample(num_rows=2000)

Change 2000 to a smaller number.

In fl_train.py, find these two lines near the top:
NUM_CLIENTS = 10
NUM_ROUNDS = 20


Change them to smaller values like
NUM_CLIENTS = 3
NUM_ROUNDS = 5


Then delete the old data files and regenerate:
rm data/synthetic_ehr.csv
rm data/ehr_metadata.json
python generate_synthetic_ehr.py

Then refer back to the running the project step.

NOTE: Reducing these values WILL affect the accuracy results. The model will still demonstrate the federated learning concept but may show lower accuracy figures.

