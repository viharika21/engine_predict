# engine_predict
Engine condtion prediction model building
git clone "https://github.com/viharika21/engine_predict.git"
cd engine_predict
## Installation:
Install all required libraries listed in requirements.txt

##execute Below commands
python model_building/train.py
python deployment/app.py

##Execution Flow:
engine_predict/
├── model_building/
│   └── train.py
├── deployment/
│   └── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .github/
    └── workflows/
        └── pipeline.yml

