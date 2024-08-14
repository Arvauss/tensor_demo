from faker import Faker
import pandas as pd
import pymongo
import random
from bson import ObjectId

fake = Faker()

num_records = 20

data = []
for _ in range(num_records):
    data.append(
        {
            "name" : fake.name(),
            "estate_id" : ObjectId("6566db6a0d3ca1493c778963"),
            "mac_address": "e4:5f:01:e7:43:69",
            "config_id" : 0,
            "geo_location": str(random.uniform(-25.9, -26.1)) + ", "  + str(random.uniform(28.1, 28.7)),
            "last_processed_infringement_date": "2024-06-05T14:44:08.338Z",
            "last_valid_processed_infringement_date": "2024-05-27T14:57:52.940Z",
            "active" : True   
        }
    )
    
df = pd.DataFrame(data)
print(data)

client = pymongo.MongoClient("mongodb+srv://dewald0725:ouu50AIZOcbOY1x2@cluster0.f9uqa08.mongodb.net/SpeedSightDB")
db = client["SpeedSightDB"]
collection = db["speed_cameras"]


collection.insert_many(df.to_dict("records"))