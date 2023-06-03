import pandas as pd
import numpy as np

# Abstract class
class ScopedAction:
    def __init__( self ) -> None:
        self.session_scope_data = dict()
        self.user_scope_data = dict()
        self.session_scope_returned = pd.DataFrame()
        self.user_scope_returned = pd.DataFrame()            
    
    def user_setup(self, user:pd.DataFrame) -> None:    
        pass    
    
    def session_run(self, session:pd.DataFrame) -> pd.DataFrame:
        return session
    
    def user_run(self, user:pd.DataFrame) -> pd.DataFrame:    
        return user
   
class CutOffAfterPremium(ScopedAction):
    def user_setup(self, user: pd.DataFrame) -> None:
        self.user_scope_data.update({"user_bought_premium": False, "break_loop": False})
    
    def session_run(self, session: pd.DataFrame) -> pd.DataFrame:
        if self.user_scope_data["user_bought_premium"]:
            self.user_scope_data.update({"break_loop": True})
            return None
        
        premium_bought_mask = session.loc[:, "event_type"] == "BUY_PREMIUM"
        bought_premium_in_session = premium_bought_mask.any()

        if bought_premium_in_session:
            time_of_buy_premium = session.loc[
                premium_bought_mask
            ].timestamp.iloc[0]
            session = session.loc[
                session.loc[:, "timestamp"] <= time_of_buy_premium
            ]
            self.user_scope_data.update({"user_bought_premium": True})
            
        return session
    

class GetAdsTime(ScopedAction):
    def user_setup(self, user: pd.DataFrame) -> pd.DataFrame:
        self.user_scope_data.update({
            "user_ads_time": np.timedelta64(0),
            "user_all_time": np.timedelta64(0)
            })
        self.user_scope_data.update({"break_loop": False})

    
    def session_run(self, session: pd.DataFrame) -> pd.DataFrame:
        self.user_scope_data["user_all_time"] += (
            session.iloc[-1, session.columns.get_loc("timestamp")] - session.iloc[0, session.columns.get_loc("timestamp")]
        )
        self.user_scope_data["user_ads_time"] += self.calculate_ads_time(session)
            
        return session
    
    def calculate_ads_time(self, df):
        ads_mask = df.loc[:, "event_type"] == "ADVERTISEMENT"
        ads_time = np.timedelta64(0)
        for _, action in df.loc[ads_mask].iterrows():
            difference = action.next_timestamp - action.timestamp
            if difference < np.timedelta64(0):
                pass
            ads_time += difference
        return ads_time
    
    def user_run(self, user: pd.DataFrame) -> pd.DataFrame:
        all_time = self.user_scope_data["user_all_time"] / np.timedelta64(1, "s")
        ads_time = self.user_scope_data["user_ads_time"] / np.timedelta64(1, "s")
        user.loc[:, "Ads_ratio"] = ads_time / all_time 
        return user

class AdsFavRatio(ScopedAction):

    def user_setup(self, user:pd.DataFrame) -> pd.DataFrame:
        ad_counts = user["event_type"].eq("ADVERTISEMENT")    
        self.user_scope_data.update({
            "adds_after_fav_count": 0,
            "all_adds_count": len(user[ad_counts].index),
            "break_loop": False
        })


    def session_run(self, session:pd.DataFrame) -> pd.DataFrame:
        session["prev_event"] = session["event_type"].shift()
        session["prev_fav_genre_track"] = session["fav_genre_track"].shift()

        ad_counts = session["event_type"].eq("ADVERTISEMENT")
        adds_after_fav_counts = len(
            session[ad_counts & session["prev_fav_genre_track"].apply(lambda x: x != [])].index
        )
        
        self.user_scope_data["adds_after_fav_count"] += adds_after_fav_counts
        return session
    
    def user_run(self, user:pd.DataFrame) -> pd.DataFrame:
        if self.user_scope_data["all_adds_count"] != 0:    
            user["adds_after_fav_ratio"] = (
            self.user_scope_data["adds_after_fav_count"] / self.user_scope_data["all_adds_count"]
            )
        elif self.user_scope_data["adds_after_fav_count"] == 0:
            user["adds_after_fav_ratio"] = np.zeros(len(user.index))
        
        user["adds_after_fav_ratio"] = user["adds_after_fav_ratio"].fillna(
            0.0
        )  # Replace NaN values with 0.0

        return user
