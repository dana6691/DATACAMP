# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv(so_survey_csv)

# Print the first five rows of the DataFrame
print(so_survey_df.head())

# Print the data type of each column
print(so_survey_df.dtypes)

# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)
################################################
#categorical variables
    #Encoding
        #1)One-hot encoding
        #2)Dummy encoding: information without deuplication
            #--> will generate too many columns, so limiting columns necessary
################################################
#One-hot encode
# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')
print(one_hot_encoded.columns)

#Create dummy variables, "DM" as a prefix 
# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=['Country'],drop_first=True, prefix='DM')
print(dummy.columns)

################################################
#Numeric variables
################################################

################################################
#categorical variables
################################################
################################################
#categorical variables
################################################
################################################
#categorical variables
################################################
################################################
#categorical variables
################################################
################################################
#categorical variables
################################################
################################################
#categorical variables
################################################
################################################
#categorical variables
################################################
################################################
#categorical variables
################################################

