#!/usr/bin/env python3
"""
Send a single message to Azure Event Hubs.
Usage: python3 send_event.py <message>
Env: EVENTHUB_CONNECTION_STRING, EVENTHUB_NAME (required when sending)
"""
import os
import sys
from azure.eventhub import EventHubProducerClient, EventData


def send_message(body: str) -> None:
    # Read connection settings from environment variables (set when the container runs).
    conn_str = os.environ.get("EVENTHUB_CONNECTION_STRING")
    eventhub_name = os.environ.get("EVENTHUB_NAME")

    # If Event Hubs is not configured, stop the program (exit with error code 1).
    if not conn_str or not eventhub_name:
        print("Event Hubs not configured (EVENTHUB_CONNECTION_STRING, EVENTHUB_NAME). Stopping.", file=sys.stderr)
        sys.exit(1)

    # Create a "producer" — the client that sends messages to Event Hubs.
    producer = EventHubProducerClient.from_connection_string(
        conn_str=conn_str,
        eventhub_name=eventhub_name,
    )

    # Send one message: put it in a batch, then send the batch.
    with producer:
        batch = producer.create_batch()
        batch.add(EventData(body=body))
        producer.send_batch(batch)


if __name__ == "__main__":
    # The message to send is the first command-line argument (e.g. "uploaded video1").
    message = sys.argv[1] if len(sys.argv) > 1 else ""
    if not message:
        print("Usage: python3 send_event.py <message>", file=sys.stderr)
        sys.exit(1)
    send_message(message)
