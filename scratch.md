# Project Notes

Notes taken while completing the [Identify Customer Segments notebook](Identify_Customer_Segments.ipynb).

## Step 1: Preprocessing

Data types:
- Most columns are either `int64` or `float64`, but 4 columns are `object` like `OST_WEST_KZ`, `CAMEO_DEUG_2015`, `CAMEO_DEU_2015`, `CAMEO_INTL_2015`.
- When converting unknown/missing data codes to NaN, I tried to convert the `missing_or_unknown` column in `feat_info` to float, but ran into errors due to the above columns containing strings like "X".
- My solution was to use `pd.map` with a lambda function to convert column cell values with NaN values if it matched any of the values in the matching `missing_or_unknown` list in the `feat_info` dataframe.
  ```py
  def convert_unknown_data_to_nan(dataframe, unknown_data_series):
      """Reads in a dataframe and maps unknown data to NaN values
      Args:
          dataframe - Pandas dataframe
          unknown_data_series - Numpy series of lists that indicate 
              missing or unknown data.
      Returns:
          Dataframe where missing or unknown data is converted to NaN
      """
      # Copy dataframe
      df_copy = dataframe.copy()
      for column, row in zip(df_copy, unknown_data_series):
          # Skip over columns where there are no codes for missing/unknown values
          if row != '[]':
              # Convert string representation of a list to an actual list
              result = row.strip('][').split(',')
              # Transform column with NaN values in place
              df_copy[column] = df_copy[column].map(
                  lambda x: np.nan if str(x) in result else x)
      return df_copy
  ```

Assess how much missing data there is in each column of dataset.

Missing values per row:
```
count    891221.000000
mean          5.460254
std          13.405008
min           0.000000
25%           0.000000
50%           0.000000
75%           1.000000
max          49.000000
```

Columns with no missing values
```
ZABEOTYP                    0
SEMIO_TRADV                 0
SEMIO_PFLICHT               0
SEMIO_KAEM                  0
SEMIO_DOM                   0
SEMIO_KRIT                  0
SEMIO_RAT                   0
SEMIO_KULT                  0
SEMIO_ERL                   0
SEMIO_LUST                  0
SEMIO_VERT                  0
SEMIO_MAT                   0
SEMIO_FAM                   0
SEMIO_SOZ                   0
SEMIO_REL                   0
FINANZ_MINIMALIST           0
ANREDE_KZ                   0
FINANZ_SPARER               0
FINANZ_VORSORGER            0
FINANZ_ANLEGER              0
FINANZ_UNAUFFAELLIGER       0
GREEN_AVANTGARDE            0
FINANZ_HAUSBAUER            0
FINANZTYP                   0
```

### Re-encoding Categorical Features

```
Columns with binary values: 
['ANREDE_KZ', 'GREEN_AVANTGARDE', 'SOHO_KZ', 'VERS_TYP', 'OST_WEST_KZ']

Columns with categorical values: 
['CJT_GESAMTTYP', 'FINANZTYP', 'GFK_URLAUBERTYP', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'SHOPPER_TYP', 'TITEL_KZ', 'ZABEOTYP', 'GEBAEUDETYP', 'WOHNLAGE', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'CAMEO_INTL_2015', 'KBA05_BAUMAX', 'PLZ8_BAUMAX']

OST_WEST_KZ: Building location via former East / West Germany (GDR / FRG) 
W    584439
O    158233
Name: OST_WEST_KZ, dtype: int64
```

Using `OneHotEncoding` on all the categories doesn't work because some of them have NaN values.  Drop or encoding into a mean value?

```py
# azdias_low[categorical].isna().sum()

CJT_GESAMTTYP                0
FINANZTYP                    0
GFK_URLAUBERTYP              0
LP_LEBENSPHASE_FEIN          0
LP_LEBENSPHASE_GROB          0
LP_FAMILIE_FEIN              0
LP_FAMILIE_GROB              0
LP_STATUS_FEIN               0
LP_STATUS_GROB               0
NATIONALITAET_KZ         30754
PRAEGENDE_JUGENDJAHRE    24034
SHOPPER_TYP              33302
TITEL_KZ                     0
ZABEOTYP                     0
GEBAEUDETYP                  0
WOHNLAGE                     0
CAMEO_DEUG_2015           3338
CAMEO_DEU_2015            3338
CAMEO_INTL_2015           3338
KBA05_BAUMAX                 0
PLZ8_BAUMAX                  0
```

Additionally, I don't know how to concatenate this onehotencoder with a pandas dataframe...

TODO: Come back to this section


### Engineer Mixed Type Features

Scratch code - tried using a for loop but manually mapping is faster:
```py
decade_40 = [1,2]
decade_50 = [3,4]
decade_60 = [5,6,7]
decade_70 = [8,9]
decade_80 = [10,11,12,13]
decade_90 = [14,15]

decades = [decade_40, decade_50, decade_60, decade_70, decade_80, decade_90]

azdias_eng = azdias_low.copy()
decade_int = []
for i, row in enumerate(azdias_eng['PRAEGENDE_JUGENDJAHRE']):
    print(row, type(row))
    if row == np.NaN:
        decade_int.append(0)
    else:
        for decade in decades:
            if row in decade:
                print(decades.index(decade), i)
                decade_int.append(decades.index(decade))
azdias_eng['decade'] = decade_int
```

1. `KBA05` fields, which are derived from the RR3 micro-cell features.
```
attribute	nan_ratio
KBA05_ANTG1	0.149597
KBA05_ANTG2	0.149597
KBA05_ANTG3	0.149597
KBA05_ANTG4	0.149597
KBA05_BAUMAX	0.534687
KBA05_GBZ	0.149597
```
2. `PLZ8` fields, which are derived from the PLZ8 macro-cell features.
```
attribute 	| nan_ratio
--- | ---
PLZ8_ANTG1  | 0.130736
PLZ8_ANTG2	| 0.130736
PLZ8_ANTG3	| 0.130736
PLZ8_ANTG4	| 0.130736
PLZ8_BAUMAX	| 0.130736
PLZ8_HHZ	  | 0.130736
PLZ8_GBZ	  | 0.130736
```


### Select and Re-encode Features

```
feat_mixed_cat['type'].value_counts()

categorical    18
mixed           5
Name: type, dtype: int64

# Count information-levels
feat_mixed_cat['information_level'].value_counts()

person           17
building          3
microcell_rr4     3
Name: information_level, dtype: int64
```

```
Features with binary values: 
['ANREDE_KZ', 'GREEN_AVANTGARDE', 'SOHO_KZ', 'VERS_TYP', 'OST_WEST_KZ']
Total binary features: 5

Features with categorical values: 
['CJT_GESAMTTYP', 'FINANZTYP', 'GFK_URLAUBERTYP', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'SHOPPER_TYP', 'ZABEOTYP', 'GEBAEUDETYP', 'WOHNLAGE', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'CAMEO_INTL_2015']
Total binary features: 18

OST_WEST_KZ: Building location via former East / West Germany (GDR / FRG) 
W    610801
O    165612
Name: OST_WEST_KZ, dtype: int64
```

Columns post filtering categorical features: 
Index(['ALTERSKATEGORIE_GROB', 'ANREDE_KZ', 'FINANZ_MINIMALIST',
       'FINANZ_SPARER', 'FINANZ_VORSORGER', 'FINANZ_ANLEGER',
       'FINANZ_UNAUFFAELLIGER', 'FINANZ_HAUSBAUER', 'GREEN_AVANTGARDE',
       'HEALTH_TYP', 'RETOURTYP_BK_S', 'SEMIO_SOZ', 'SEMIO_FAM', 'SEMIO_REL',
       'SEMIO_MAT', 'SEMIO_VERT', 'SEMIO_LUST', 'SEMIO_ERL', 'SEMIO_KULT',
       'SEMIO_RAT', 'SEMIO_KRIT', 'SEMIO_DOM', 'SEMIO_KAEM', 'SEMIO_PFLICHT',
       'SEMIO_TRADV', 'SOHO_KZ', 'VERS_TYP', 'ANZ_PERSONEN', 'ANZ_TITEL',
       'HH_EINKOMMEN_SCORE', 'WOHNDAUER_2008', 'ANZ_HAUSHALTE_AKTIV',
       'ANZ_HH_TITEL', 'KONSUMNAEHE', 'MIN_GEBAEUDEJAHR', 'OST_WEST_KZ',
       'BALLRAUM', 'EWDICHTE', 'INNENSTADT', 'GEBAEUDETYP_RASTER',
       'ONLINE_AFFINITAET', 'KBA13_ANZAHL_PKW', 'ARBEIT', 'ORTSGR_KLS9',
       'RELAT_AB'],
      dtype='object')
Number of columns before: 63
Number of columns after:  45


### Engineer Mixed-type Features

Below is how the numbers are mapped to values in `CAMEO_INTL_2015`:

> German CAMEO: Wealth / Life Stage Typology, mapped to international code
> 
> - -1: unknown
> - 11: Wealthy Households - Pre-Family Couples & Singles
> - 12: Wealthy Households - Young Couples With Children
> - 13: Wealthy Households - Families With School Age Children
> - 14: Wealthy Households - Older Families & Mature Couples
> - 15: Wealthy Households - Elders In Retirement
> - 21: Prosperous Households - Pre-Family Couples & Singles
> - 22: Prosperous Households - Young Couples With Children
> - 23: Prosperous Households - Families With School Age Children
> - 24: Prosperous Households - Older Families & Mature Couples
> - 25: Prosperous Households - Elders In Retirement
> - 31: Comfortable Households - Pre-Family Couples & Singles
> - 32: Comfortable Households - Young Couples With Children
> - 33: Comfortable Households - Families With School Age Children
> - 34: Comfortable Households - Older Families & Mature Couples
> - 35: Comfortable Households - Elders In Retirement
> - 41: Less Affluent Households - Pre-Family Couples & Singles
> - 42: Less Affluent Households - Young Couples With Children
> - 43: Less Affluent Households - Families With School Age Children
> - 44: Less Affluent Households - Older Families & Mature Couples
> - 45: Less Affluent Households - Elders In Retirement
> - 51: Poorer Households - Pre-Family Couples & Singles
> - 52: Poorer Households - Young Couples With Children
> - 53: Poorer Households - Families With School Age Children
> - 54: Poorer Households - Older Families & Mature Couples
> - 55: Poorer Households - Elders In Retirement
> - XX: unknown

We can map this into 2 features with each integer representing the ordered lists:

- `wealth`: The wealth stage of the household
    1. Wealthy
    2. Propserous
    3. Comfortable
    4. Less Affluent
    5. Poorer
    
- `life_stage`: The life stage of the household
    1. Pre-Family Couples & Singles
    2. Young Couples With Children
    3. Families With School Age Children
    4. Older Families & Mature Couples
    5. Elders In Retirement

We will map the first integer (ie. values in multiple of tens) to the `wealth` feature, and the second integer to the `life_stage` feature.

Notes:
wealth
- poorest at top, 28% of distribution
- followed by prosperous, 24%
- generally shows a strasified distribution between the wealth and poorer households.

life_stage
- pre-family at the highest 30.7%
- closely followed by older families & mature couples, 29.5%
- least represented is young couples with children, 9.8%


### Apply Feature Scaling
```
HEALTH_TYP              2.59
VERS_TYP                2.59
movement                1.69
decade                  1.69
KBA13_ANZAHL_PKW        1.54
ANZ_HAUSHALTE_AKTIV     0.78
life_stage              0.73
wealth                  0.73
RELAT_AB                0.50
ARBEIT                  0.50
ORTSGR_KLS9             0.49
ANZ_HH_TITEL            0.45
ALTERSKATEGORIE_GROB    0.34
BALLRAUM                0.07
EWDICHTE                0.07
INNENSTADT              0.07
KONSUMNAEHE             0.01
OST_WEST_KZ             0.00
FINANZ_HAUSBAUER        0.00
SEMIO_REL               0.00
```

Average ratio of missing values for all rows
```py
df.isna().mean(axis=1).mean().round(6) * 100
# 0.3%
```

# Examine ratio of missing values for all rows
```py
df.isna().mean(axis=1).sort_values(ascending=False).head(20)

# 673695    0.122449
# 340685    0.122449
# 485886    0.122449
# 798560    0.122449
# 731159    0.122449
# ...
```

Sum of rows with missing values
54479 

Percentage of rows with missing values
7.02%

Total cells with missing values: 115027 

Percentage of cells with missing values: 0.3 %

1. SEMIO_KAEM       
2. ANREDE_KZ       
3. SEMIO_DOM       
4. SEMIO_KRIT      
5. FINANZ_HAUSBAUER

1. EMIO_SOZ        
2. EMIO_FAM        
3. EMIO_KULT       
4. INANZ_MINIMALIST
5. EMIO_VERT       
   
   
##### Component 4 Interpetations

Below are the mappings of the features from the data dictionary:

##### Most Positive Features:

1. GREEN_AVANTGARDE - Membership in environmental sustainability as part of youth (0: not a member of green avantgarde, 1: member of green avantgarde)
2. movement - Political alignment of the person during their youth (0: Mainstream, 1:Avantgarde)
3. EWDICHTE - Density of households per square kilometer (1: less than 34 households per km^2, 6: more than 999 households per km^2)
4. ORTSGR_KLS9 - Size of community (1: <= 2,000 inhabitants, 9: > 700,000 inhabitants)
5. ONLINE_AFFINITAET - Online affinity (0: none, 5: highest)

##### Most Negative Features:

1. FINANZ_HAUSBAUER - Financial typology (1: very high, 5: very low)
2. wealth - The wealth stage of the household (1. Wealthy, 5. Poorer)
3. INNENSTADT - Distance to city center (downtown) (1: in city center, 8: more than 40 km to city center)
4. BALLRAUM - Distance to nearest urban center (1: less than 10 km, 7: more than 100 km)
5. HH_EINKOMMEN_SCORE - Estimated household net income (1: highest income, 6: very low income)

##### Interpretation

The least represented segment are those that are affiiliated with activist movements (GREEN_AVANTGARDE, movement), that live in an urban environment (EWDICHTE, ORTSGR_KLS9) and have a high online affinity (ONLINE_AFFINITAET). Based on the negative features, they have a higher socio-economic status (FINANZ_HAUSBAUER, wealth, HH_EINKOMMEN_SCORE) and live near the city center (INNENSTADT, BALLRAUM). 
   
