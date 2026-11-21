"""Podgląd wyniku: python form2json/show.py <plik.fields.json>"""
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print(d['title'],'|',d['text_source']); 
for f in d['fields']: print(f"{f['detector'][:12]:12} {f['type']:11} {f['label'][:58]!r:60} {f.get('options','')} {f.get('max_length','')} | {f.get('section','')[-35:]}")
