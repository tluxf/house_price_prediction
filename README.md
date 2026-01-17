# House Price Prediction
An end-to-end machine learning project. House price and demographic data sourced from publicly accessable datasets, three machine learning models tuned and trained, and the model predictions assessed. Full writeup is availabel at (link here).

## Importing and Processing data
Raw .csv files were imported, or Excel files downloaded and processed to form .csv files for the source data. Raw data was processed to produce the train and test datasets for the models in the Jupyter Notebooks 01_Process train.ipynb and 01_Process test.ipynb respectively. Additional functions used in the processing pipeline are saved in helper_functions.py.

Raw data was obtained from the following sources:

House sale price data: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

LSAO, MSAO and LTLA for postcodes: https://geoportal.statistics.gov.uk/datasets/c4f84c38814d4b82aa4760ade686c3cc/about

Latitude and longitude for postcodes: https://www.freemaptools.com/download-uk-postcode-lat-lng.htm

Demographic data: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates

Commute data: https://www.ons.gov.uk/datasets/TS058/editions/2021/versions/4/filter-outputs/1242b10f-061d-4db7-9e69-ab1f2036e00f#get-data

Wage data: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/smallareaincomeestimatesformiddlelayersuperoutputareasenglandandwales

## Train Data Exploration
The test data was thoroughly examined in the Jupyter notebook 02_Data exploration.ipynb

## Model Preparation, Training and Implimentation
Three machine learning models were created and trained using the data from the train dataset, and used to predict prices using data from the test dataset using the sklearn package. A linear regression model in 03_Linear model.ipynb, a k-nearest neighbour model in  03_KNN model.ipynb and a random forest model in 03_Random forest model.ipynb. Additional functions used in the model preparation are available in the file helper_functions_training.py.

## Model Assessment
The predicted prices for the test dataset from the three models were assessed and compared in the file 04_Assessing predictions.ipynb.
