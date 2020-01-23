import requests
import string
from nltk.tokenize import word_tokenize

Output: ['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']
#Anadir la URL de la pagina para obtener las definiciones
resp = requests.get('http://api.urbandictionary.com/v0/define?term=O_o')
res = resp.json()
if len(res["list"]) != 0:
    first_def = res["list"][0]["definition"]
    first_def_wo_definition = first_def.translate(str.maketrans('', '', string.punctuation))
    tokenized_definition = word_tokenize( first_def_wo_definition)
    print(tokenized_definition)
else:
    print("No definition")

    