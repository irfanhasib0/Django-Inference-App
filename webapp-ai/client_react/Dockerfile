FROM node:alpine as node_cient
WORKDIR /app
COPY ./ ./
RUN mv ./libs ../
RUN rm -rf node_modules
RUN npm install -g npm@8.14.0
RUN npm install 

#RUN sleep 5h
#RUN npm audit fix --force
#RUN ls node_modules/react-scripts/bin/
#RUN node_modules/react-scripts/bin/react-scripts.js start
#CMD ["npm" , "start"]
#CMD ["npm", "run", "start"]
