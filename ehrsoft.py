from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import pandas as pd
import os


app = Flask(__name__)
#req for sessions/login (need to change in production)
app.secret_key = '2468otu3p' 

#session timeout set to 10minutes of inactivity
app.config['PERMANENT_SESSION_LIFETIME'] =timedelta(minutes=10)

#storing passwords as hashes
#generate_password_hash takes a plain password and turns it into a long scrambled string using PBKDF2-SHA256
USERS = {
	'doctor': generate_password_hash('password123'),
	'admin': generate_password_hash('adminpass')
}

#write timestamp entry to auditlog file
def write_audit_log(event):
	import datetime
	#create log folder if it doesnt exist
	os.makedirs('logs', exist_ok=True)

	#open log file in append mode
	with open('logs/audit.log', 'a') as f:
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		f.write(f"[{timestamp}] {event}\n")



@app.route('/', methods=['GET', 'POST'])
def login():
	#route handles both showing login page and processing form subm.
	if request.method == 'POST':
		username = request.form.get('username')
		password = request.form.get('password')

		#comparing entered password against stored hash
		if username in USERS and check_password_hash(USERS[username], password):
			#making session last (so the 10min timeout applies)
			session.permanent = True
			#store username in session
			session['user'] = username
			write_audit_log(f"Login success - user: {username}")
			flash('Login successful!', 'success')
			return redirect(url_for('patients'))
		else:
			#recording failed attempt
			write_audit_log(f"Login failed - attempted username: {username}")
			flash('Invalid credentials', 'danger')

	return render_template('login.html')

@app.route('/patients')
def patients():
	#checking if user is logged in before showing patient data. if not logged in, redirect them to login page
	if 'user' not in session:
		flash('Please log in first', 'warning')
		return redirect(url_for('login'))

	csv_path = 'data/synthetic_ehr.csv'
	if not os.path.exists(csv_path):
		return "No synthetic data found. Run generate_synthetic_ehr.py first."

	df = pd.read_csv(csv_path)

	#show 100 rows (make number less if encountering performance issue)
	table_html = df.head(100).to_html(classes='table table-striped', index=False, border=0)
	return render_template('patients.html', table=table_html, username=session['user'])

@app.route('/train')
def train():
	#only logged in users can use training feature
	if 'user' not in session:
		return redirect(url_for('login'))

	flash('Starting Federated Learning training across 10 simulated hospital silos (20 rounds) - (this may take some time)', 'info')

	try:
		#import and run the federated learning script
		import fl_train
		#the training will run here
		fl_train.run()
		#record successful training run
		write_audit_log(f"FL training completed - triggered by user: {session['user']}")
		flash('Federated Learning completed successfully!', 'success')
	except Exception as e:
		#log the error and show error message
		write_audit_log(f"FL training failed - user: {session['user']} - error: {str(e)}")
		flash(f'Error during training: {str(e)}', 'danger')

	return redirect(url_for('patients'))


@app.route('/logout')
def logout():
	write_audit_log(f"Logout - user: {session.get('user', 'unknown')}")
	session.pop('user', None)
	flash('Logged out successfully', 'success')
	return redirect(url_for('login'))

if __name__ == '__main__':
#host='0.0.0.0' makes it accessible from your host machine if needed
    app.run(debug=True, host='0.0.0.0', port=5000)  
