import pandas as pd
# Geography Package
from geopy.geocoders import Nominatim

filename = 'data.xlsx'

df_deliver_from = pd.read_excel(filename, sheet_name='from')  # Shape (32776, 14)
df_deliver_to = pd.read_excel(filename, sheet_name='to')
df_deliver_to_notnull = df_deliver_to[df_deliver_to.HasNAs == 0] # Shape (657, 14) 339/657 missing some data

geolocator = Nominatim(user_agent="delivery", timeout=10)

lats = []
lngs = []

for address in df_deliver_to['FullAddress']:
    print(address)
    if geolocator.geocode(address) is None:
        print('ping not received')
        lats.append('NaN')
        lngs.append('NaN')

    else:
        print('ping received')
        location =  geolocator.geocode(address)
        lats.append(location.latitude)
        lngs.append(location.longitude)
#        df_deliver_to['lat'] = location.latitude
#        df_deliver_to['lng'] = location.longitude

df_deliver_to['lat'] = lats
df_deliver_to['lng'] = lngs



#Tim start here

# Below hashed out line needs to be fixed. Basically last dataframe for df_deliver_to, get rid of the rows that have no lat lngs
#df_deliver_to = df_deliver_to[(df_deliver_to.lng != 'NaN') | (df_deliver_to.lat != 'NaN')]    

df_delivery = df_deliver_from.merge(df_deliver_to, how='inner', left_on='Account', right_on='Account') # Shape (40619, 27)

# Above df is the main table that needs to be in a database

# import sqlalchemy, create an engine for SQLlite, connection to it

# make a table from dataframe pass through the engine and connection

# Make a test query command to see if you are able to retreive the data back to terminal



#After fixing that, this is the 