# STEP 1: Pull python image
FROM python:3.9.13

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /challenge1
COPY requirements.txt /challenge1

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -U --no-cache-dir gdown --pre
RUN pip install gdown

# RUN pip install virtualenv 
# RUN virtualenv aiqn
# RUN source aiqn/bin/activate
RUN pip install -r requirements.txt

# STEP 7: Download file weight if needed
# RUN gdown "https://drive.google.com/uc?export=download&id=1VIplhJoaKPI08Qcdq6FhPk_dMvGOfDMp"

# STEP 8: RUN COMMAND
COPY . /challenge1
CMD ["python", "api.py"]
