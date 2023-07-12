install:
	python.exe -m pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv test_*.py
	
format:
	black *.py

lint:
	pylint --disable=R,C *.py

all: install lint format test

apprun: 
	python app.py

trainmodel:
	python train.py --model_name st