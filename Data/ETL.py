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
        print('-- ping not received')
        lats.append('NaN')
        lngs.append('NaN')

    else:
        print('++ ping received')
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

#Import SQLite
import sqlite3 as db
#Establish Connection
conn = db.connect('data.db') 
#Create a cursor to execute SQL
sql = conn.cursor()
#Push DF to SQL
df_delivery.to_sql(name='delivery',con=conn,if_exists='replace')
conn.commit()
#query
delivery_df_final = sql.execute("""SELECT * FROM delivery""")
conn.commit()


######################################################
# Back-Up Method with sqlalchemy and engine where columns
# deleted due to bad data types

#Delete columns with bad data types
#del df_delivery['PlanArrival']
#del df_delivery['ActualArrival']

# import sqlalchemy, create an engine for SQLlite, connection to it
#import sqlalchemy as db
#from sqlalchemy import create_engine
#engine = create_engine('sqlite:///data.db', echo=False)

# make a table from dataframe pass through the engine and connection
#df_delivery.to_sql('data', con=engine,if_exists='replace',index=False)
# Make a test query command to see if you are able to retreive the data back to terminal

#query = "SELECT * FROM data"
#query_df = pd.read_sql(query,engine)
