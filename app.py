# Age column fix for date_of_birth
if 'date_of_birth' in df.columns:
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['Age'] = 2025 - df['date_of_birth'].dt.year
elif 'YOB' in df.columns:
    df['Age'] = 2025 - df['YOB']
elif 'Year_Birth' in df.columns:
    df['Age'] = 2025 - df['Year_Birth']
    
   
