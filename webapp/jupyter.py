import subprocess
ret = str(subprocess.run("docker exec -it runner bash -ic 'jupyter lab list'",shell=True,capture_output=True).stdout)
print(ret)
start_ind = ret.find('token=') + 6
ret = ret[start_ind:]
end_ind = ret.find(' ')
print('http://localhost:9003/?token='+ret[:end_ind])

