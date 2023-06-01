import pandas as pd
import numpy as np
import os
from typing import List
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from scoped_action import (
    ScopedAction,
    CutOffAfterPremium,
    GetAdsTime,
    AdsFavRatio,
)
from utils.fast_reader import FastReader

mlb = MultiLabelBinarizer(sparse_output=True)
lb = LabelBinarizer()


class DataModel(object):
    def __init__(
        self,
        load_data: bool = True,
        data_paths_dict: dict = {
            "users_path": "../data_jsonl/users.jsonl",
            "tracks_path": "../data_jsonl/tracks.jsonl",
            "artists_path": "../data_jsonl/artists.jsonl",
            "sessions_path": "../data_jsonl/sessions.jsonl",
        },
    ):
        self.users_path = data_paths_dict["users_path"]
        self.tracks_path = data_paths_dict["tracks_path"]
        self.artists_path = data_paths_dict["artists_path"]
        self.sessions_path = data_paths_dict["sessions_path"]
        self.users_df = pd.DataFrame()
        self.tracks_df = pd.DataFrame()
        self.artists_df = pd.DataFrame()
        self.sessions_df = pd.DataFrame()
        if load_data:
            self.load_data()

    def load_data(self):
        self.users_df = FastReader.read_json(
            self.users_path,
        )
        self.tracks_df = FastReader.read_json(self.tracks_path)
        self.artists_df = FastReader.read_json(self.artists_path)
        self.sessions_df = FastReader.read_json(self.sessions_path)
        self.sessions_df.loc[:, "timestamp"] = pd.to_datetime(
            self.sessions_df.loc[:, "timestamp"]
        )

        self.tracks_df.rename(columns={"id": "track_id"}, inplace=True)
        self.artists_df.rename(columns={"id": "id_artist"}, inplace=True)
        return self

    def get_merged_dfs(self, N=None):
        all_df = self.users_df[
            ["user_id", "premium_user", "city", "favourite_genres"]
        ].merge(self.sessions_df, on="user_id")
        all_df = all_df.merge(
            self.tracks_df[["track_id", "id_artist"]], on="track_id", how="left"
        )
        all_df = all_df.merge(
            self.artists_df[["id_artist", "genres"]], on="id_artist", how="left"
        )

        all_df["genres"] = all_df["genres"].fillna("").apply(list)
        return all_df if N is None else all_df.iloc[:N, :]


class Preprocessor:
    # @staticmethod
    # def register_session_scoped_action(name:str, user_scope_function, session_scope_function):
    #     scoped_actions.append(CutOffAfterPremium(name, user_scope_function, session_scope_function))

    @staticmethod
    def preprocess_scoped(
        sessions_df: pd.DataFrame, scoped_actions: List[ScopedAction] = None
    ):
        # problem with static class - this cant be class property or default argumentent value
        scoped_actions = (
            [CutOffAfterPremium(), GetAdsTime(), AdsFavRatio()]
            if scoped_actions is None
            else scoped_actions
        )

        df_filtered = pd.DataFrame()
        for user_id, user_actions in sessions_df.groupby("user_id"):
            user_filtered = pd.DataFrame()

            # Setup for all actions
            for scoped_action in scoped_actions:
                scoped_action.user_setup(user_actions)

            for session_id, session in user_actions.groupby("session_id"):
                if all(
                    [state.user_scope_data["break_loop"] for state in scoped_actions]
                ):
                    break

                session = Preprocessor.set_next_timestamp(session)
                session = Preprocessor.set_fav_genre_track(session)
                for scoped_action in scoped_actions:
                    session = scoped_action.session_run(session)
                    if session is None:
                        break

                user_filtered = pd.concat([user_filtered, session])

            # Run for all actions - after each session was run
            for scoped_action in scoped_actions:
                user_filtered = scoped_action.user_run(user_filtered)

            df_filtered = pd.concat([df_filtered, user_filtered])

        return df_filtered

    @staticmethod
    def set_next_timestamp(session) -> pd.DataFrame:
        # returns sorted ndarray with next_timestamp column regarding the session scope
        session.sort_values(by=["timestamp"])
        pd.options.mode.chained_assignment = None
        session.loc[:, "next_timestamp"] = session.loc[:, "timestamp"].shift(
            -1, fill_value=session.loc[:, "timestamp"].max()
        )
        return session

    @staticmethod
    def set_fav_genre_track(session) -> pd.DataFrame:
        session["fav_genre_track"] = session.apply(
            lambda row: list(set(row["favourite_genres"]) & set(row["genres"])), axis=1
        )
        return session

    @staticmethod
    def get_event_type_count_df(df):
        all_events_counts = (
            df.groupby("user_id").size().reset_index(name="all_events_count")
        )
        event_type_count = (
            df.groupby(["user_id", "event_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        event_type_count = event_type_count.merge(all_events_counts, on="user_id")
        event_types = ["user_id"] + [
            "event_type_" + col for col in event_type_count.columns[1:-1]
        ]
        event_type_count.columns = event_types + ["all_events_count"]
        for col in event_types[1:]:
            event_type_count[col] = (
                event_type_count[col] / event_type_count["all_events_count"]
            )
        event_type_count.drop("event_type_BUY_PREMIUM", axis="columns", inplace=True)
        event_type_count.drop("all_events_count", axis="columns", inplace=True)
        return event_type_count

    @staticmethod
    def run(df, drop_user_id=True):
        df = df.join(
            pd.DataFrame(
                lb.fit_transform(df.pop("city")),
                index=df.index,
                columns=lb.classes_,
            )
        )

        df = Preprocessor.preprocess_scoped(df)

        event_type_count_df = Preprocessor.get_event_type_count_df(df)
        df = pd.merge(df, event_type_count_df, on=["user_id"])

        # one hot encoding
        df = df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df.pop("favourite_genres")),
                index=df.index,
                columns=mlb.classes_,
            )
        )

        # Ważna notatka: dropnięcie user_id powinno być opcjonlane - bo w metodzie User.to_vector jakoś trzeba
        # wyciągnąć rekordy z interesującym nas userem

        to_drop = [
            "timestamp",
            "track_id",
            "event_type",
            "session_id",
            "id_artist",
            "genres",
            "next_timestamp",
            "fav_genre_track",
            "prev_event",
            "prev_fav_genre_track",
        ]

        to_drop = to_drop + ["user_id"] if not drop_user_id else to_drop

        for col in to_drop:
            df.drop(col, axis=1, inplace=True)

        return df