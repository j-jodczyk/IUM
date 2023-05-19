import pandas as pd

class ScopedAction:
    # Would be best in maintanance if w e had context for each scope - session, user, ...
    # class Context:
    #     def __init__(self, data:list=None) -> None:
    #         self.data = data
            
    def __init__( self ) -> None:
        self.session_scope_data = dict()
        self.user_scope_data = dict()
        self.break_loop = False
        self.user_loop = False
        self.session_scope_returned = pd.DataFrame()
        self.user_scope_returned = pd.DataFrame()            
            
    def session_run(self, session:pd.DataFrame) -> pd.DataFrame:
        pass
    
    def user_run(self, user:pd.DataFrame) -> pd.DataFrame:    
        pass    
    
    def user_add_data(self, args: list[dict]):
        for arg in args:
            self.user_scope_data.update( arg )
    
    def session_add_data(self, args: list[dict]):
        for arg in args:
            self.session_scope_data.update( arg )
    
class CutOffAfterPremium(ScopedAction):
    def user_run(self, user: pd.DataFrame) -> pd.DataFrame:
        self.user_scope_data.update({"user_bought_premium": False})
    
    def session_run(self, session: pd.DataFrame) -> pd.DataFrame:
        if self.user_scope_data["user_bought_premium"]:
            self.break_loop = True
            return    
        
        session.sort_values(by=["timestamp"])
        premium_bought_mask = session.loc[:, "event_type"] == "BUY_PREMIUM"
        bought_premium_in_session = premium_bought_mask.any()

        if bought_premium_in_session:
            time_of_buy_premium = session.loc[
                premium_bought_mask
            ].timestamp.iloc[0]
            session = session.loc[
                session.loc[:, "timestamp"] <= time_of_buy_premium
            ]
            self.user_scope_data["user_bought_premium"] = True
            
        return session