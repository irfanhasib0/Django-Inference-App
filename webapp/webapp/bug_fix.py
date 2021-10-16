import re

with open('webapp/registry.py','r') as file:
     data = file.read()
with open('/usr/local/lib/python3.8/site-packages/django/apps/registry.py','w') as file:
     file.write(data)
 