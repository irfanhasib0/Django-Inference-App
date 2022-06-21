FROM node:16-bullseye

ENV NODE_ENV=production

COPY . /usr/local/share/node
WORKDIR /usr/local/share/node
RUN npm install .
RUN rm -rf ~/.npm ~/.cache

CMD ["node", "index.js"]
