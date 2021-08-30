import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'aceleracion':0.199713, 'velocidad':4.323565})

print(r.json())