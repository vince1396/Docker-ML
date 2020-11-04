FROM ubuntu:latest

WORKDIR /workspace

COPY ../Docker-ML /workspace

RUN apt-get -y update \
    && apt-get install -y python3 python3-pip vim git \
    && git config --global user.name vince1396 \
    && git config --global user.email vincent.cotini1996@gmail.com \
    && git clone https://github.com/vince1396/ML-Hotel-booking.git \
    && pip3 install -r requirements.txt

EXPOSE 8000

CMD ["jupyter", "notebook", "--port=8000", "--no-browser", "--allow-root", "--ip=0.0.0.0"]
