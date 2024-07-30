from query_understanding import *
from categories_recommendation import *
from items_database import *
from reranking import *
from helper import *
from heapq import nlargest

class RecommendationPipeline():
    def __init__(self):
        self.query_understanding = QueryUnderstandingModule(device="cpu", embedding_size=512)
        self.category_recommendation = CategoryRecommendationModule()
        self.database = ItemsDatabaseModule()
        self.scoring = RerankingModule(device="cpu")
        
    def recommend(self, table_name, item_image, n_items, k=2):
        outfits = []
        candidated_combos = []
        if item_image is None: return None
        embedding, main_category = self._query_understanding(item_image)
        recommended_categories = self._recommend_categories(main_category, n_items-1, k)
        for category_list in recommended_categories:
            # Query database for similar items
            candidate_items_dict = self._query_database_multiple_categories(table_name, embedding, category_list)
            candidated_combos.extend(self._generate_combinations(candidate_items_dict))
        best_combos = self._get_best_combos(embedding, candidated_combos, k)
        for best_combo in best_combos:
            outfit = [(category + "/" + image_path, category) for image_path, category, _ in best_combo]
            outfits.append(outfit)
        self.close()
        return outfits
        
    def get_category(self, image):
        _, category = self._query_understanding(image)
        return category
    
    # Step 1: Understanding the item
    def _query_understanding(self, image):
        image = preprocess_image(image)
        if image is None: return None
        embedding, category = self.query_understanding.forward(image)
        return embedding, category
    
    # Step 2: Get recommended categories
    def _recommend_categories(self, category, length, k):
        recommended_categories_list = self.category_recommendation.recommend_categories(category, length, k)
        return recommended_categories_list
    
    # Step 3: Query similar images
    ## First, we write a function to query for each category
    def _query_database_one_category(self, table_name, image_embedding, category, n_items=10):
        query = f"""
            SELECT image, category, embedding
            FROM {table_name}
            WHERE category = '{category}'
            ORDER BY embedding <-> '{add_commas(image_embedding)}'
            LIMIT {n_items}
        """
        results = self.database.query(query)
        return results
    
    ## Then, we write a function to query for multiple categories
    def _query_database_multiple_categories(self, table_name, image_embedding, categories, n_items=10):
        candidate_items_dict = {}
        for category in categories:
            candidate_items_dict[category] = self._query_database_one_category(table_name, image_embedding, category, n_items=n_items)
        return candidate_items_dict
    
    def _generate_combinations(self, candidate_items_dict):
        combos = list(product(*candidate_items_dict.values()))
        return combos
    
    def _get_best_combos(self, image_embedding, combos, k):
        scored_combos = []
        for combo in combos:
            embeddings = [embedding for _, _, embedding in combo]
            embeddings.append(image_embedding)
            score = self.scoring.score(embeddings)
            scored_combos.append((score, combo))
        best_combos = nlargest(k, scored_combos, key=lambda x: x[0])
        return [combo for _, combo in best_combos]
    
    def close(self):
        self.database.close()
