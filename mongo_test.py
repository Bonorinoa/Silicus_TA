from pymongo.mongo_client import MongoClient
import datetime as dt
import numpy as np

uri = "mongodb+srv://augusto:Antartica2023!?@silicusta.jfzl5zt.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp"

# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    # database and collection code goes here
    db = client.Silicus_TA_DB
    coll = db.Feedback
    
    feedback = "This is a test feedback"
    messages = "This is a test message"
    total_cost = 5
    time_spent = 122.0
    
    
    # insert code goes here
    feedback_doc = {"Random_id": np.random.randint(10000),
                     "Content": {
                                "feedback": feedback,
                                "chat_history": messages,
                                "total_cost": total_cost,
                                "time_until_feedback": time_spent,
                                "submission_date": dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                                }
                     }

    result = coll.insert_one(feedback_doc)
    print("Inserted these docs: ", result.inserted_ids)

    # find code goes here
    #cursor = coll.find({"orbitalPeriod": 6.41})

    # iterate code goes here
    #for doc in cursor:
    #    print(doc)
        
    # Close the connection to MongoDB when you're done.
    client.close()
except Exception as e:
    print(e)
    # Close the connection to MongoDB when you're done.
    client.close()
    
    
