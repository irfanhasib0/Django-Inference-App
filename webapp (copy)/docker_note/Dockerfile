FROM node:slim as note_app

ENV NODE_ENV=production

COPY ./sticky-notes/ /usr/local/share/sticky-notes/
WORKDIR /usr/local/share/sticky-notes
#RUN sleep 5h
RUN npm install --force.
RUN rm -rf ~/.npm ~/.cache
CMD ["node", "server/index.js"]
