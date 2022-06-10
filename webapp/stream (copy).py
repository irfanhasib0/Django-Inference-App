import subprocess
ret = subprocess.call("'docker exec -it runner bash -ic 'jupyter lab list'",stdout=subprocess.PIPE)
print(ret.stdout)

