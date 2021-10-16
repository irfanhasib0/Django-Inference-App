#docker-compose run web django-admin startproject webapp .
#docker-compose run web python manage.py startapp api_1
#docker exec -it webapp_web_1 bash -ic 'python3 manage.py startapp api_1'
#docker exec -it webapp_web_1 bash -ic 'python3 manage.py createsuperuser'
#docker exec -it webapp_web_1 bash -ic 'python3 manage.py makemigrations'
#docker exec -it webapp_web_1 bash -ic 'python3 manage.py migrate'
#apt install libgl1-mesa-glx
docker exec -it webapp_web_1 bash -ic 'sed -i s/"raise RuntimeError(\"populate() isn't reentrant\")"/"self.app_configs = {}"/g /usr/local/lib/python3.8/site-packages/django/apps/registry.py'