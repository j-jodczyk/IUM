import pandas as pd
import numpy as np

# Abstract class
class ScopedAction:
    # Would be best in maintanance if w e had context for each scope - session, user, ...
    # class Context:
    #     def __init__(self, data:list=None) -> None:
    #         self.data = data
            
    def __init__( self ) -> None:
        self.session_scope_data = dict()
        self.user_scope_data = dict()
        # TODO: refactor - make signals class for
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
        # TODO: can be changed to assign those values for the session instead of summing 
        self.user_scope_data["user_all_time"] += (
            session.iloc[-1, session.columns.get_loc("timestamp")]
            - session.iloc[0, session.columns.get_loc("timestamp")]
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
