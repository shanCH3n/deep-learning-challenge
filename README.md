# deep-learning-challenge
The objective of this project is to employ deep learning techniques, such as neural networks, to forecast the success rate of organizations financed by Alphabet Soup Charity. The focus is on constructing a binary classification model that can effectively determine whether an organization will achieve success or not, by analyzing a range of features.

## Background
Alphabet Soup Charity operates as a charitable entity that finances philanthropic ventures globally. The non-profit receives contributions from diverse origins and subsequently dispenses the funds to other charitable institutions. To guarantee optimal utilization of the funds, Alphabet Soup Charity seeks to establish a predictive model that can forecast an organization's likelihood of success based on specific features.

Alphabet Soup's business team has provided a CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organisation, such as:

- EIN and NAME: Identification columns
- APPLICATION_TYPE: Alphabet Soup application type
- AFFILIATION: Affiliated sector of industry
- CLASSIFICATION: Government organisation classification
- USE_CASE: Use case for funding
- ORGANIZATION: Organisation type
- STATUS: Active status
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Special considerations for application
- ASK_AMT: Funding amount requested
- IS_SUCCESSFUL: Was the money used effectively

## Data Preprocessing
Our model's target variable is the IS_SUCCESSFUL column, representing an organization's success status. All other columns, except for EIN and NAME, which serve as identification columns and hold no significance in our analysis, form the model's features. There were 9 such features: 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', 'ASK_AMT'. 

To preprocess the data before creating the model, we undertook the following measures:

- Checked for missing values and duplicates within dataset
- Dropped the EIN and NAME columns.
- Consolidated low-frequency values in the APPLICATION_TYPE and CLASSIFICATION columns into an Other category.
- Converted categorical data to numeric through the use of the pd.get_dummies function (One-hot encoding)


## Training & Optimising our Model
Using the Keras library, a deep learning model with 2 hidden layers, each with a varying number of neurons and activation functions selected via the Keras Tuner library. The output layer comprises 1 neuron using a sigmoid activation function.

The model utilized the 'Adam' optimizer, an optimization algorithm that uses both momentum and adaptive learning rates to speed up the convergence of the optimization process, and the binary cross-entropy loss function. The dataset was first split into training and testing sets and a StandardScaler was used to normalize the data. During training, the ModelCheckpoint callback function was utilised to save the model weights every 5 epochs.

On testing, the model (Model 1) attained an accuracy of 72.55%. Unfortunately, this did not meet our anticipated model performance of 75%. To enhance the model's accuracy, we experimented with altering the number of neurons and layers, changing the activation functions, and changing. However, these efforts did not yield a considerable improvement in accuracy (Model 2: 72.65% Accuracy).

## Recommendations
TBC

