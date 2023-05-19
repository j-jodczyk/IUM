import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

from src.microservice.scoped_action import ScopedAction, CutOffAfterPremium

mlb = MultiLabelBinarizer(sparse_output=True)
lb = LabelBinarizer()
class DataModel( object ):
    def __init__(self, load_data: bool=True, data_paths_dict: dict = {"../data/users_path": "users.json", "../data/tracks_path":"tracks.json", "../data/artists_path":"artists.json", "../data/sessions_path":"sessions.json"}):
        self.users_path=data_paths_dict["users_path"]
        self.tracks_path=data_paths_dict["tracks_path"]
        self.artists_path=data_paths_dict["artists_path"]
        self.sessions_path=data_paths_dict["sessions_path"]
        self.users_df=pd.DataFrame()
        self.tracks_df=pd.DataFrame()
        self.artists_df=pd.DataFrame()
        self.sessions_df=pd.DataFrame()
        if load_data:
            self.load_data()
    
    def load_data(self):
        self.users_df = pd.read_json(self.users_path)
        self.tracks_df = pd.read_json(self.tracks_path)
        self.artists_df = pd.read_json(self.artists_path)
        self.sessions_df = pd.read_json(self.sessions_path)

        self.tracks_df.rename(columns={"id": "track_id"}, inplace=True)
        self.artists_df.rename(columns={"id": "id_artist"}, inplace=True)


class Preprocessor:
    # @staticmethod
    # def register_session_scoped_action(name:str, user_scope_function, session_scope_function):
    #     Preprocessor.scoped_actions.append(CutOffAfterPremium(name, user_scope_function, session_scope_function))

    scoped_actions = [ CutOffAfterPremium() ]
    
    @staticmethod
    # def preprocess_scoped(self):
    def cut_off_after_buy_premium( sessions_df:pd.DataFrame, scoped_actions: list[ScopedAction] = scoped_actions):
        sessions_filtered = pd.DataFrame()
        for user_id, user_actions in sessions_df.groupby("user_id"):
            if all([state.break_loop for state in Preprocessor.scoped_actions]):
                break
            
            for scoped_action in Preprocessor.scoped_actions:
                scoped_action.user_run(user_actions)

            for session_id, session in user_actions.groupby("session_id"):
                for scoped_action in Preprocessor.scoped_actions:
                    session = scoped_action.session_run(session)

                sessions_filtered = pd.concat([sessions_filtered, session])
        return sessions_filtered


    @staticmethod
    def calculate_ads_time(df):
        ads_mask = df.loc[:, "event_type"] == "ADVERTISEMENT"
        ads_time = np.timedelta64(0)
        for _, action in df.loc[ads_mask].iterrows():
            difference = action.next_timestamp - action.timestamp
            if difference < np.timedelta64(0):
                pass
            ads_time += difference
        return ads_time

    @staticmethod
    def get_adds_time_df(sessions_df):
        time_comparison_df = pd.DataFrame()

        for user_id, user_actions in sessions_df.groupby("user_id"):
            user_ads_time = np.timedelta64(0)
            user_all_time = np.timedelta64(0)

            for session_id, session in user_actions.groupby("session_id"):
                # sanity sort - should be sorted by now anyways
                session.sort_values(by=["timestamp"])
                user_all_time += (
                    session.iloc[-1, session.columns.get_loc("timestamp")]
                    - session.iloc[0, session.columns.get_loc("timestamp")]
                )
                # TODO maybe investigate this chain warning a bit more instead of supressing:
                # https://towardsdatascience.com/how-to-suppress-settingwithcopywarning-in-pandas-c0c759bd0f10
                pd.options.mode.chained_assignment = None
                session.loc[:, "next_timestamp"] = session.loc[:, "timestamp"].shift(
                    -1, fill_value=session.loc[:, "timestamp"].max()
                )

                user_ads_time += Preprocessor.calculate_ads_time(session)

            session_times_df = pd.DataFrame(
                {
                    "all_time": user_all_time / np.timedelta64(1, "s"),
                    "ads_time": user_ads_time / np.timedelta64(1, "s"),
                },
                index=[user_id],
            )
            time_comparison_df = pd.concat([time_comparison_df, session_times_df])
        return time_comparison_df

    def get_merged_dfs(self):
        all_df = self.users_df[["user_id", "favourite_genres"]].merge(
            self.sessions_df, on="user_id"
        )
        all_df = all_df.merge(
            self.tracks_df[["track_id", "id_artist"]], on="track_id", how="left"
        )
        all_df = all_df.merge(
            self.artists_df[["id_artist", "genres"]], on="id_artist", how="left"
        )

        all_df["genres"] = all_df["genres"].fillna("").apply(list)
        return all_df

    # TODO refector if time - better structure
    def get_ads_after_fav_ratio_df(self):
        all_df = self.get_merged_dfs()
        all_df["fav_genre_track"] = all_df.apply(
            lambda row: list(set(row["favourite_genres"]) & set(row["genres"])), axis=1
        )

        # TODO: do the rest inside sessions:

        all_df["prev_event"] = all_df["event_type"].shift()
        all_df["prev_fav_genre_track"] = all_df["fav_genre_track"].shift()

        ad_counts = all_df["event_type"].eq("ADVERTISEMENT")
        adds_after_fav_counts = (
            all_df[ad_counts & all_df["prev_fav_genre_track"].apply(lambda x: x != [])]
            .groupby("user_id")
            .size()
            .rename("adds_after_fav_count")
        )
        all_adds_counts = (
            all_df[ad_counts].groupby("user_id").size().rename("all_adds_count")
        )

        final_df = pd.DataFrame(
            {
                "adds_after_fav_count": adds_after_fav_counts,
                "all_adds_count": all_adds_counts,
            }
        ).reset_index()
        final_df["adds_after_fav_ratio"] = (
            final_df["adds_after_fav_count"] / final_df["all_adds_count"]
        )
        final_df["adds_after_fav_ratio"] = final_df["adds_after_fav_ratio"].fillna(
            0.0
        )  # Replace NaN values with 0.0
        final_df = final_df[["user_id", "adds_after_fav_ratio"]]

        return final_df

    def get_event_type_count_df(self):
        all_events_counts = (
            self.sessions_df.groupby("user_id")
            .size()
            .reset_index(name="all_events_count")
        )
        event_type_count = (
            self.sessions_df.groupby(["user_id", "event_type"])
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

    def run(self):
        self.df = self.df.join(
            pd.DataFrame(
                lb.fit_transform(self.df.pop("city")),
                index=self.df.index,
                columns=lb.classes_,
            )
        )

        self.sessions_df = self.cut_off_after_buy_premium()

        # Unify naming "adds" and "ads"
        time_comparison_df = Preprocessor.get_adds_time_df(self.seesion_df)
        self.df = self.df.join(
            pd.DataFrame(
                data=time_comparison_df.loc[:, "ads_time"]
                / time_comparison_df.loc[:, "all_time"],
                columns=["Ads_ratio"],
            ),
            on="user_id",
        )

        event_type_count_df = self.get_event_type_count_df()
        self.df = pd.merge(self.df, event_type_count_df, on=["user_id"])

        adds_after_fav_ratio_df = self.get_ads_after_fav_ratio_df()
        self.df = pd.merge(self.df, adds_after_fav_ratio_df, on=["user_id"])

        # one hot encoding
        self.df = self.df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(self.df.pop("favourite_genres")),
                index=self.df.index,
                columns=mlb.classes_,
            )
        )

        return self.df
