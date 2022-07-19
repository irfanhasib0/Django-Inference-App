#!/bin/bash
sudo docker cp stream.py runner:/stream.py
sudo docker exec -it runner bash -ic 'ls /'
sudo docker exec -it runner bash -ic 'chmod +x /stream.py && python3 /stream.py'
