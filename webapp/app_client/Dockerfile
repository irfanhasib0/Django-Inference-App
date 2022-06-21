FROM node:alpine
WORKDIR /app
COPY package.json ./
#COPY package-lock.json ./
COPY ./ ./
#RUN npm install ./src/components/editor_cnt/
#RUN sleep 5h
RUN npm install --save react-draft-wysiwyg draft-js react-draft-wysiwyg-a
RUN npm i
CMD ["npm", "run", "start"]
