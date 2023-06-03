from kafka import KafkaConsumer, TopicPartition

# Configure the Kafka consumer
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')

# Get the list of topics
topic_partitions = consumer.topics()

# Print the list of topics
for topic in topic_partitions:
    print(topic)

# Close the consumer
consumer.close()