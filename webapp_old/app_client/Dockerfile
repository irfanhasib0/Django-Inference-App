FROM node:alpine
WORKDIR /app
COPY package.json ./
#COPY package-lock.json ./
COPY ./ ./
#RUN npm install ./src/components/editor_cnt/
#RUN sleep 5h
RUN npm install --save draft-js
RUN npm install --save ./libs/react-draft-wysiwyg-local-a
RUN npm install --save ./libs/draftjs-utils
RUN npm install --save ./libs/html-to-draftjs
#RUN npm install --save react-draft-wysiwyg draft-js react-draft-wysiwyg-a
RUN npm install --save react-canvas-draw
RUN npm install --save react-sortablejs sortablejs
RUN npm install --save sass
RUN npm i
CMD ["npm", "run", "start"]
