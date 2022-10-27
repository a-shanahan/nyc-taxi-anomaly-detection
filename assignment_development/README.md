# Taxi Driver Assignment
Assignment of taxi drivers to customers is usually left to human intuition or a set of 
business rules. Business rules can be effective but can struggle in dynamic environments. 

Instead, a model trained via reinforcement is used to assign taxi drivers to customer orders. This is 
obviously a fictional scenario but is used for demonstration purposes. 

The fictitious company is focused on maximising profits as quickly as possible but balances this 
against customer service and ensuring a fair distribution of work.

## Setup
### Initial Data

### Environment

```shell
python3 -m venv venv
source venv/bin/activate
```

## Model Training
To train the model:

```shell
python3 model_train.py
```

## Model Evaluation:

```shell
python3 evaluate.py
```

## 