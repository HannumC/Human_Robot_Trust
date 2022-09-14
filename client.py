import socket

def client(message):
	host = socket.gethostname()  # get local machine name
	port = 5000  # Make sure it's within the > 1024 $$ <65535 range
  
	s = socket.socket()
	s.connect((host, port))

	s.send(message.encode('utf-8'))
	data = s.recv(1024).decode('utf-8')
	print('Received from server: ' + data)
	s.close()

if __name__ == '__main__':
	client('the message')