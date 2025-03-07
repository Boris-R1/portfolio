# Portfolio
## 1_Titanic
A classic machine learning problem of classifying the Titanic's passengers.
https://www.kaggle.com/c/titanic/overview

Considered multiple models with different parameters.  
Utilised principal component analysis.  
RandomForestClassifier provided the best results with the accuracy of 0.775.

## 2_Credit_card_fraud_detection
The task was to develop a machine learning model to detect fraudulent credit card transactions using anonymized transaction data. 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Tested Logistic Regression capabilities in credit card fraud detection and found optimal parameters. 
Compared different approached to combating class imbalance: manual undersampling, manual oversampling, ADASYN, SMOTE, etc.

## 3_Twitter_sentiment_analysis
The objective of this task was to detect hate speech in tweets.
https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech

Tested several approached to twitter sentiment analysis: Naive Bayes Classifier with the Bag of Words, Naive Bayes Classifier with TF-IDF, Logistic Regression with TF-IDF and Matrix Dimensionality Reduction. 
The highest recall was 0.874 for precision > 0.4 (TF-IDF classifier, MultinomialNB) and 0.717 for precision > 0.5 (TF-IDF classifier, BernoulliNB).
Compared performance of ADASYN and SMOTE in the task.

## 4_Animal_face_classification
The objective was to develop a custom neural network to classify images of animal faces into their respective categories.
https://www.kaggle.com/datasets/andrewmvd/animal-faces

Built a small neural network consisting of such layers as Conv2d, BatchNorm, Dropout, Activation, MaxPool2d, Linear.
Selected optimal parameters and obtained accuracy of 0.922 with a high potential for further improvement.

## 5_Face_expression_recognition
The objective was to fine-tune an existing neural network in order to classify facial expressions into different emotional categories based on image data.
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

Fine-tuned ResNet-50 Neural Network to recognise face expressions with the highest accuracy of 0.704 obtained.
Tested Dynamic Quantization.
Applied Knowledge Distillation (ResNet-50 as a teacher, MobileNetV3-Small as a student). Param size dropped by 24.5 times, estimated total size decreased by 7.8 times, inference time decreased by 3.2 times, accuracy went to 0.683 (3% difference).

