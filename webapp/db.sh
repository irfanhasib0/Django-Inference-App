sudo docker exec -it webapp_db_mysql_1 bash -ic 'mysql -u MYSQL_USER -pMYSQL_PASSWORD notes'
#CREATE USER 'MYSQL_USER'@'localhost' IDENTIFIED BY 'MYSQL_PASSWORD';
#GRANT CREATE,SELECT,DELETE,INSERT,UPDATE ON notes * TO MYSQL_USER@localhost; FLUSH PRIVILEGES;
#SELECT * FROM user_database WHERE user = user_1 AND topic = topic_1 AND section = 0
#INSERT INTO user_database (user, topic, section, title, content) VALUES (99, '', '', '', '');


