Run dataset_generate.py in order to generate the full dataset that will be used to train and test the neural network.

Once you run dataset_generate.py, you will see a file called my_training_data_noGP_3ep.npz inside the same directory as the script. 

Next run main.py and make sure my_training_data_noGP_3ep.npz is inside the same folder. The script main.py will train a neural network with dropout layers first and plot predicted outputs with uncertainty values. It also contains code for training an ensemble of neural networks that are used to generate uncertainty values. Theses values are also plotted to a graph.