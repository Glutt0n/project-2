# Stroke Prediction
 The goal of this is to help the stakeholders understand in what way the data does or does not help to predict a stroke in patients.
## KNeighborsClassifier vs. LogisticRegression Models

**Author**:Leard Russell 

### Business problem:

Can we predict whether a patient is likely to get a stroke when given info on:
'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status'. 

### Data:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download (Confidential Source)

![image](https://user-images.githubusercontent.com/118066797/232075317-4e17c999-59c6-497f-be6c-d5215c9af198.png)



![image](https://user-images.githubusercontent.com/118066797/231963179-c2b32ca7-d8d7-489c-878c-4aee975b765a.png)
  
**Insights:**

This plot shows a positive correlation between age and average bmi. The data shows that average bmi increase as age does. Of all features, 'age' has the strongest correlation with our target, 'stroke'. In a way, this correlation demonstrates that having a higher BMI increases the risk of having a stroke.

![image](https://user-images.githubusercontent.com/118066797/231963252-9c6f8ed8-a472-4fb7-8597-5c4f7722f9f0.png)
 **Insights:**

This plot shows that patients with presumably higher stressful occupations are also older on average. This means that there is an indirect link from 'work_type' and our 'stroke' target. The previously shown heatmap shows the highest correlation between age and risk of stroke. 

   


### Methods:
- Data preparation steps with explanation and justification for choices
- Data Cleaning: getting rid of duplicates rows and incorrect/inconsistent values
- Validation Split: splitting data to prepare it for machine learning
- Data Transformation: Using transformers and pipelines for preprocessing
- Hyperparameter Tuning Using GridSearchCV: 
    KNN - n_neighbors, p value, weights
    LogisticRegression - solver, penalty, C value
- Feature Extraction with PCA
- Evaluating a KNeighbors Classifier model and a LogisticRegression model's ability to predict stroke using multiclass classification report **while prioritizing recall scores.** 


#### Best Model Metrics:
**(Default) KNeighbors Classifier:**
• Results for training data:
  - Accuracy = 0.96
  - Precision = 0.03
  - Recall = 0.06
  
• Results for test data:
  - Accuracy = 0.95
  - Precision = 0.03
  - Recall = 0.06


If I were to choose a model of the two utilized, I would have choose the default KNeighborsClassifier for the data.
I'd choose this due to the model having the highest of all metrics, but most importantly the highest recall. The model with the highest recall, in the context of the data, is going to be the one with the least amount of false negatives. Less false negatives means less patients deemed "likely to get stroke." 

When comparing both our KNeighborsClassifier model and our LogisticRegression model, after tuning multiple hyperparameters for each using GridSearchCV, and after completing feature extraction using PCA we see that there is no significant change in metrics, especially in our poorest performing metrics, recall and precision. In fact, our recall scores in our test set actually dropped from 0.06 in the default KNN Model to 0.03 after hyperperamater tuning. After applying PCA with 10 components, it dropped to 0.

One major takeaway from the data is that there simply isn't high enough correlation between individual features and between features and our 'stroke' target for any model to demonstrate any predictive ability. This is evident in the extremely low recall scores(our most important metrics for the data). With low recall scores like these, there would be a dangerously large amount of patients that wouldn't be considered at risk for stroke when they actually are. 

## Recommendations:

• Considering all models with and without tuning, if one absolutely had to be implemented, I would have to recommend using the default KNeighbors Classifier simply due to it having the *highest* recall scores.

• Another strong recommendation to stakeholders is to gather more data with features that at the very least provide a moderate correlation to our 'stroke' target so that a machine learning model can actually learn from the data. Both the KNN and LogReg model had poor metrics even after having multiple hyperparameters tuned in addition to feature extraction using PCA. This demonstrates a lack of predictive more so in the data than from the models. More *quality* data is needed. 

For any additional questions, please contact **leardrussell@gmail.com**
