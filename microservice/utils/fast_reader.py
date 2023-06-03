import json
import pandas as pd 
 
class FastReader(object):
    @staticmethod
    def read_json(json_path:str):
        with open(json_path) as f:
            lines = f.read().splitlines()
                
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']
        
        df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
        
        return df_final
