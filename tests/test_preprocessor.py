import unittest
import pandas as pd
from src.microservice.load_data import Preprocessor


class PreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.sessions_df = pd.read_json("./src/data/sessions.json")
        # self.sessions_df = pd.read_json("../src/data/sessions.json")

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
                self.assertEqual(user_sessions.iloc[-1]["event_type"], "BUY_PREMIUM")
