import json
import time
from kafka import KafkaProducer
import pandas as pd

producer = KafkaProducer(
    bootstrap_servers='localhost:9092'
)

df = pd.read_csv('online.csv')
df = df.drop('Diabetes_binary', axis=1)

for idx, row in df.iterrows():
    record = row.to_dict()

    print(json.dumps(record))

    producer.send(
        topic="health_data",
        value=json.dumps(record).encode("utf-8")
    )

    time.sleep(0.5)

producer.flush()
producer.close()
print("Done!")