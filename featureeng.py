import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset=pd.read_csv('student_dropout_dataset_v3.csv')


categorical_cols = dataset.select_dtypes(include=['object']).columns
numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.drop('Student_ID')


# 3. Encode Categorical Data 
# Note: We keep NaNs as NaNs during encoding so the Imputer can see them
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Only encode non-null values to preserve the 'missing' status for the imputer
    series = dataset[col]
    valid_mask = series.notnull()
    dataset.loc[valid_mask, col] = le.fit_transform(series[valid_mask])
    le_dict[col] = le

# 4. Scale the data
# Distance calculations fail if one variable has a much larger range than others
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(dataset.drop(columns=['Student_ID'])), 
                         columns=dataset.columns.drop('Student_ID'))

# 5. Apply KNN Imputer
# n_neighbors=5 is standard; weights='distance' gives closer neighbors more influence
imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed_array = imputer.fit_transform(df_scaled)

# 6. Convert back to DataFrame and Inverse Scale
df_final = pd.DataFrame(df_imputed_array, columns=df_scaled.columns)
df_final = pd.DataFrame(scaler.inverse_transform(df_final), columns=df_final.columns)

# 7. Convert Categorical columns back to original labels (and round them)
for col in categorical_cols:
    df_final[col] = df_final[col].round().astype(int)
    # Map back to original strings if needed
    # df_final[col] = le_dict[col].inverse_transform(df_final[col])

print("Missing values after imputation:", df_final.isnull().sum().sum())

df_final.to_csv('df_final.csv')
#lets go

#NICE