# Chatbot Deployment with Flask

In this Project, I built the chatbot using pytorch and deploy with Flask.

## Initial Setup:

This repo currently contains the starter files.

Clone repo and create a virtual environment

```
$ git clone https://github.com/yves137/chatbot-backend.git
$ cd chatbot-backend
$ python3 -m venv cbot
$ . cbot/bin/activate
```

Install dependencies

```
$ (cbot) pip install -r requirements.txt
```

Install nltk package

```
$ (cbot) python
>>> import nltk
>>> nltk.download('punkt')
```

Modify `intents.json` with different intents and responses for your Chatbot

Run

```
$ (cbot) python train.py
```

This will dump data.pth file. And then run
the following command to test it in the console.

```
$ (cbot) python chat.py
```
