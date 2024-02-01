# About
This is a question answering model (for multiple choices quizzes), made for the ninth task of SemEval-2024: [BRAINTEASER](https://brainteasersem.github.io/).

Differ from the normal multiple choices model provided by huggingface, which has a fixed number of choices (usually 4), this model can have an arbitrary number of choices by leveraging the vector representation of each choice in its prediction.

# Results
Well, the result is quite disappointing (bellow is the models accuracies along with its ranking on the leaderboard, only the encoder is used for T5 model):

Word Puzzle:
pretrained model|W_ori|W_sem|W_con|W_ori_sem|W_ori_sem_con|W_overall
---|---|---|---|---|---|---
google/flan-t5-xl|0.844 (6)|0.750 (8)|0.781 (7)|0.750 (8)|0.625 (10)|0.792 (11)
google/flan-t5-large| 0.781 |0.812 | 	0.812 | 	0.719 | 	0.625 | 	0.802

Sentence Puzzle:
pretrained model|S_ori|S_sem|S_con|S_ori_sem|S_ori_sem_con|S_overall
---|---|---|---|---|---|---
google/flan-t5-xl|0.725 (11)|0.800 (9)|0.750 (9)|0.675 (14)|0.525 (16)|0.758 (19)
google/flan-t5-large|0.725 |0.725 | 	0.700 | 	0.675 | 	0.550 | 	0.717

This poor results could probably be due to the model overfitting on the small training dataset (500 entries for sentence puzzle and 400 for word puzzle, trained separately) and the 2.1 millions newly initialized parameters (8.4 mils for xl model).

Further experimentation reveals that turning the data augmentation off decreases the model performance by 15% (as expected) but has an opposite effect on smaller model (10% increase on google/flan-t5-small). And the miniscule change in result when switching to a larger model (from large to xl as shown above) or the fact that the google/flan-t5-xl model used for Sentence Puzzle submission was only fine-tuned for 1 epoch are signs of overfitting. Ergo, fine-tunning google/flan-t5-xl directly would have yield better results on this small training dataset as this approach will perform better on larger dataset.

# Training

This project required python 3.11 (or 3.10) which can be installed using pyenv or conda. Then follow the steps laid out in `run.ipynb` notebook.

