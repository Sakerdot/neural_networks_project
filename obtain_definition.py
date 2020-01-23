import requests
import string
from nltk.tokenize import word_tokenize

def obtain_definition(word):
    url = "http://api.urbandictionary.com/v0/define?term=" + word
    resp = requests.get(url)
    res = resp.json()
    if len(res["list"]) != 0:
        first_def = res["list"][0]["definition"]
        first_def_wo_definition = first_def.translate(str.maketrans('', '', string.punctuation))
        tokenized_definition = word_tokenize( first_def_wo_definition)
        return tokenized_definition
    else:
        return []

# a = obtain_definition("O_o")
# print(a)
