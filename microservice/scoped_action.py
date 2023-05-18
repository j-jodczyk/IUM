import pandas as pd

class ScopedAction:
    # Would be best in maintanance if w e had context for each scope - session, user, ...
    # class Context:
    #     def __init__(self, data:list=None) -> None:
    #         self.data = data
            
    def __init__( self, name:str, 
                    user_scope_function: function=None, session_scope_function: function=None,
                        
                        ) -> None:
        self.name = name
        self.user_scope_function = user_scope_function
        self.session_scope_function = session_scope_function
        
        self.session_scope_data = dict()
        self.user_scope_data = dict()
        
        self.session_scope_returned = pd.DataFrame()
        self.user_scope_returned = pd.DataFrame()            
        
    def session_add_data(self, args: list(dict)):
        for arg in args:
            self.session_scope_data.update( arg )
        
    def session_run(self, session:pd.DataFrame):
        if self.session_scope_function is not None:
            self.session_scope_returned.append(self.for_session_function(session))
        
    def user_add_data(self, args: list(dict)):
        for arg in args:
            self.user_scope_data.update( arg )
    
    def user_run(self, user:pd.DataFrame):    
        if self.session_scope_function is not None:
            self.user_scope_returned.append(self.for_user_function(user))
    
    
class CutOffAfterPremium(ScopedAction):
    def user_run(self, user: pd.DataFrame):
        self.user_scope_data.update({"user_bought_premium": False})
        return super().user_run(user)
    
    def session_run(self, session: pd.DataFrame):
        if user_bought_premium:
            break
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
            user_bought_premium = True

        return super().session_run(session)