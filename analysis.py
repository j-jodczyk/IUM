import pandas as pd

sessions_df = pd.read_json("sessions.jsonl")
users_df = pd.read_json("users.jsonl")

# print(sessions_df.info())
# print(users_df.info())

# print(sessions_df.head())

print(sessions_df.value_counts(sessions_df["event_type"]))
