## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import boxcox
data = pd.read_csv('Encoding Data.csv')
print("Original Dataset:")
print(data.head())

data.fillna(data.mean(numeric_only=True), inplace=True)
numeric_column = data.select_dtypes(include=np.number).columns[0]
print(f"\nColumn Selected for Transformation: {numeric_column}")
positive_data = data[data[numeric_column] > 0].copy()
positive_data['Log_Transform'] = np.log(positive_data[numeric_column])
positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]
positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])
positive_data['Square_Transform'] = np.square(positive_data[numeric_column])
positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])
print(f"\nBox-Cox Lambda Value: {lambda_value}")
pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])
scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])
positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)

print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())
```
# RESULT:
<img width="335" height="207" alt="553863038-72a2ff47-1080-44a5-a9b3-c8d8ad4fc4c8" src="https://github.com/user-attachments/assets/e196db83-4052-486e-9b69-1a9a100dc2bd" />
<img width="716" height="463" alt="553864034-99aa8bf3-977c-47cd-bba0-731553c5d652 (1)" src="https://github.com/user-attachments/assets/fd5e98d7-b7ca-4ee6-8cc7-5071c1bf6994" />

       
