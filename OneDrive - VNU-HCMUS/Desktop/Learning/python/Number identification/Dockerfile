
FROM ubuntu

USER root  

# copy necessary file 
COPY /processing ./processing 
COPY mnist_data.npz .
COPY requirements.txt .

# update apt-get package that ensures this latest package
RUN apt-get update

# install python and create virtual environment which will serve in this project
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get install -y python3-venv
RUN python3 -m venv venv
ENV PATH="/venv/bin/activate"

# install necessary labries
RUN apt-get install --no-cache-dir -r requirements.txt

# run code s
CMD [ "python3" ,"./processing/predict.py", "2" ]


