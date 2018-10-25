FROM centos:7
  
# Install necessary packages
RUN yum install -y epel-release
RUN yum install -y centos-release-scl
RUN yum install -y rh-python36
RUN scl enable rh-python36 bash
RUN yum install -y python36-setuptools
RUN easy_install-3.6 pip
RUN pip3 install --upgrade pip

RUN yum install -y libgomp

RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed numpy
RUN pip3 install -U setuptools
RUN pip3 install pillow flask-socketio eventlet pandas sklearn keras scikit-image imutils tensorflow lightgbm opencv-python
RUN yum install -y libXext libSM libXrender

RUN pip3 install --upgrade lightgbm==2.1.2
RUN pip3 install tensorflow==1.5

RUN mkdir /log
RUN scl enable rh-python36 bash
# Copy the current directory contents into the container at /app
RUN mkdir /app


  
# Set the working directory to /app
WORKDIR /app
ADD . /app
# Run sample_bot.py when the container launches, you should replace it with your program
ENTRYPOINT ["python3.6", "sample_bot.py"]
