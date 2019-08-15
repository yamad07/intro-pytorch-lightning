from models.cnn import CNN
import os
from pytorch_lightning import Trainer
from test_tube import Experiment

model = CNN()
exp = Experiment(save_dir=os.getcwd())

trainer = Trainer(experiment=exp, max_nb_epochs=1, train_percent_check=0.1)
trainer.fit(model)
