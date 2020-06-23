import smtplib
from email.mime.text import MIMEText

def sendMessage(subject, message=''):
	msg = MIMEText(message)

	msg['Subject'] = subject
	msg['From'] = 'xxx'
	msg['To'] = 'xxx'

	smtpObj = smtplib.SMTP_SSL('xxx', 465)
	smtpObj.login('xxx', 'xxx')
	smtpObj.send_message(msg)
	smtpObj.quit()

def sendFile(subject, fileName, message=''):
	with open(fileName) as fp:
		msg = MIMEText(message+"\n"+fp.read())

	msg['Subject'] = subject
	msg['From'] = 'xxx'
	msg['To'] = 'xxx'

	smtpObj = smtplib.SMTP_SSL('xxx', 465)
	smtpObj.login('xxx', 'xxx')
	smtpObj.send_message(msg)
	smtpObj.quit()

