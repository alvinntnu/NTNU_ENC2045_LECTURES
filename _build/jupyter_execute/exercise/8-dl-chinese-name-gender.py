# Assignment VIII: Deep Learning

## Question 1

Use the dataset, `DEMO_DATA/chinese_name_gender.txt` and create a Chinese name gender classifier using the deep learning method. You need to include a few important considerations in the creation of the deep learning classifer.

1. Please consult the lecture notes and experiment with different architectures of neural networks. In particular, please try combinations of the following types of network layers:

    - dense layer
    - embedding layer
    - RNN layer
    - bidirectional layer

2. Please include regularizations and dropbouts to avoid the issue of overfitting.
3. Please demonstrate how you find the optimal hyperparameters for the neural network using `keras-tuner`.
4. Please perform post-hoc analyses on a few cases using `LIME` for more interpretive results.



plot_model(model1, show_shapes=True)

plot_model(model2, show_shapes=True)

plot_model(model3)

plot_model(model4)

plot_model(model5)

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['Male'], char_level=True)

exp = explainer.explain_instance(
X_test_texts[text_id], model_predict_pipeline, num_features=100, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'陳宥欣', model_predict_pipeline, num_features=100, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'李安芬', model_predict_pipeline, num_features=2, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'林月名', model_predict_pipeline, num_features=2, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'蔡英文', model_predict_pipeline, num_features=2, top_labels=1)
exp.show_in_notebook(text=True)