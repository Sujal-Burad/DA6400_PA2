## Getting Started

Before running the project, ensure you have all necessary dependencies installed by executing the following command in your terminal:

pip install -r requirements.txt



## Running Dueling-DQN (type1, and type2) and MC-REINFORCE(with and without baseline):
To run the algorithms:
1. Navigate into the corresponding algorithm directory.
2. Execute the `main.py` file using Python with command arguments and change the gym environment inside as needed.


## Hyperparameter Tuning

For hyperparameter tuning of the algorithms:

1. Open the corresponding '.py' file.
2. Modify the parameters you wish to fine-tune.
3. Adjust the sweep method as required. (We used the 'bayes' sweep method as it is typically more efficient than grid or random search. It uses past results to inform future searches, focusing on the most promising areas of the hyperparameter space.

