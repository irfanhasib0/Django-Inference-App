FROM node:alpine
WORKDIR /app
COPY package.json ./
COPY package-lock.json ./
COPY ./ ./
RUN npm i
#RUN mysql -h mysql_db -u MYSQL_USER -pMYSQL_PASSWORD books < /db/setup.sql
CMD ["npm", "run", "start"]
