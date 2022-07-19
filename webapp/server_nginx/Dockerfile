#FROM nginx:stable-alpine
FROM ubuntu:20.04
#RUN apt install nginx
#sudo /etc/init.d/nginx start
RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
#RUN apt -y upgrade
#RUN add-apt-repository universe
#RUN apt -y install nginx 
RUN apt -y install wget git gcc g++
RUN apt-get install -y git build-essential ffmpeg libpcre3 libpcre3-dev libssl-dev zlib1g-dev
RUN git clone https://github.com/sergey-dryabzhinsky/nginx-rtmp-module.git
RUN wget http://nginx.org/download/nginx-1.17.6.tar.gz
RUN tar -xf nginx-1.17.6.tar.gz
WORKDIR nginx-1.17.6
RUN ./configure --prefix=/etc/nginx --with-http_ssl_module --add-module=../nginx-rtmp-module
RUN make -j 1
RUN make install
#COPY ./webapp_nginx.conf /etc/nginx/sites-available/
#COPY ./webapp_nginx.conf /etc/nginx/conf.d/default.conf
#COPY ./usgi_params /
#RUN mkdir /etc/nginx/sites-enabled
#RUN ln -s /etc/nginx/sites-available/webapp_nginx.conf /etc/nginx/sites-enabled/
#RUN chmod +x /etc/init.d/nginx
#RUN sleep 5h
#RUN export PATH=$PATH:/usr/local/nginx/sbin
RUN chmod +x /etc/nginx/sbin/nginx
RUN /etc/nginx/sbin/nginx
WORKDIR /
