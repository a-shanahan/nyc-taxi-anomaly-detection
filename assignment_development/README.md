# Taxi Driver Assignment
Assignment of taxi drivers to customers is usually left to human intuition or a set of 
business rules. Business rules can be effective but can struggle in dynamic environments. 

Instead, a model trained via reinforcement is used to assign taxi drivers to customer orders. This is 
obviously a fictional scenario but is used for demonstration purposes. 

The fictitious company is focused on maximising profits as quickly as possible but balances this 
against customer service and ensuring a fair distribution of work.

## Setup
### Initial Data Generation
The [data_prep.py](data_prep.py) script generates a number of synthetic datasets that are used 
to initialse the model training environment. Location IDs have been previously converted to 
latitude/longitude values for use in calculating distance.

### Environment

Create a virtual environment and install the required Python libraries:

```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Model Training
To train the model:

```shell
python3 model_train.py
```

The model was trained using a Google Colab GPU enabled notebook. Training using an M2 Apple GPU was attempted 
but failed when a large number of training steps was specified. 

## Model Evaluation:

```shell
python3 evaluate.py
```

To evaluate the effectiveness of the model, a number of simulations are run and assessed to see how 
many steps it takes to reach the required profit. Also of interest is the mean pickup time and utilisation 
spread at the end of the simulation.

[Evaluate_Data.ipynb](Evaluate_Data.ipynb) has been provided to read in the output view the results. 
