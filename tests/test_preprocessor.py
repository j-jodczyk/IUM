import unittest
import pandas as pd
import numpy as np
from src.microservice.load_data import Preprocessor, CutOffAfterPremium, DataModel


class PreprocessorTest(unittest.TestCase):
    def setUp(self):
        # self.sessions_df = pd.read_json("./src/data/sessions.json")
        self.sessions_df = pd.read_json("../src/data/sessions.json")

    def test_cut_off_after_premium(self):
        # GIVEN
        cutter = CutOffAfterPremium()
        buy_premium = "BUY_PREMIUM"
        # WHEN
        for user_id, user in self.sessions_df.groupby(by="user_id"):
            cutter.user_setup(user)
            for session_id, session in user.groupby(by="session_id"):
                if all([state.user_scope_data["break_loop"] for state in [cutter]]):
                    break

                size_before = len(session.index)
                session = Preprocessor.set_next_timestamp(session)
                session = cutter.session_run(session)

        # THEN
                if session is None:
                    break
                
                if np.any(session.loc[:, "event_type"] == buy_premium):
                    self.assertEqual(session.iloc[-1]["event_type"], buy_premium)
                else:
                    self.assertEqual(len(session.index), size_before)
                                        
            cutter.user_run(user)
            
    def test_cut_off(self):
        # GIVEN
        # sessions_df rows containing BUY_PREMIUM events
        bought_premium_mask = self.sessions_df.loc[:, "event_type"] == "BUY_PREMIUM"
        bought_premium = self.sessions_df[bought_premium_mask]
        # list of all premium users
        premium_users = bought_premium[["user_id"]].values.T.tolist()[0]
        
        # WHEN
        cut_off_df = Preprocessor.cut_off_after_buy_premium(self.sessions_df)

        # THEN
        # The last action is BUY_PREMIUM 
        for user_id, user_sessions in cut_off_df.groupby("user_id"):
            if user_id in premium_users:
                user_sessions.sort_values(by="timestamp", inplace=True)
                self.assertEqual(user_sessions.iloc[-1]["event_type"], "BUY_PREMIUM")

    def test_run(self):
        data_model = DataModel()
        df = data_model.get_merged_dfs()
        Preprocessor.run(df)