from __init__ import *

class CategoryRecommendationModule():
    def __init__(self, rules_path=RULES_PATH):
        self.rules_df = pd.read_csv(rules_path)
    
    def recommend_categories(self, category, length, k):
        # Get rows where self.rules_df[category] equals 1
        rows = self.rules_df.loc[self.rules_df[category] == 1]
        count_ones = rows.sum(axis=1)
        # Get rows that count_ones equals length + 1
        row_indices = rows.loc[count_ones == (length + 1)].drop_duplicates().index.to_list()
        # For each row, get the column names and store in a seperate list for each combo
        recommended_categories = []
        for index in row_indices:
            new_list = self.rules_df.columns[self.rules_df.loc[index] == 1].to_list()
            # Remove category from new_list
            new_list.remove(category)
            recommended_categories.append(new_list)
        return recommended_categories[:k]