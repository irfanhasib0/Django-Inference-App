#docker-compose run web django-admin startproject webapp .
#docker-compose run web python manage.py startapp api_1
#docker exec -it web bash -ic 'python3 ~/manage.py createsuperuser'
docker exec -it webapp_web_1 bash -ic 'python3 ~/manage.py startapp api_1'