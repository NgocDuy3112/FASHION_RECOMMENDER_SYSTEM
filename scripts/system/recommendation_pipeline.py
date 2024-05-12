from query_understanding import *
from categories_recommendation import *
from items_database import *
from scoring import *
from helper import *

class RecommendationPipeline():
    def __init__(self):
        self.query_understanding = QueryUnderstandingModule(device="mps", embedding_size=512)
        self.category_recommendation = CategoryRecommendationModule()
        self.database = ItemsDatabaseModule()
        self.scoring = ScoringModule(device="cpu")
        
    def recommend(self, item_image, k=2):
        outfits = []
        # item_image = preprocess_image(item_image)
        if item_image is None: return None
        embeddings, category = self._query_understanding(item_image)
        for length in range(2, 5):
            recommended_categories = self._recommend_categories(category, length, k)
            for category_list in recommended_categories:
                # Query database for similar items
                candidate_items_dict = self._query_database_multiple_categories(embeddings, category_list, k)
                combos = self._generate_combinations(candidate_items_dict)
                best_combo = self._get_best_combo(embeddings, combos)
                outfit = [category + "/" + image_path for image_path, category, _ in best_combo]
                outfits.append(outfit)
        return outfits
        
        
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
    def _query_database_one_category(self, image_embedding, category, k):
        query = f"""
            SELECT image, category, embedding
            FROM items
            WHERE category = '{category}'
            ORDER BY embedding <-> '{add_commas(image_embedding)}'
            LIMIT {k}
        """
        results = self.database.query(query)
        return results
    
    ## Then, we write a function to query for multiple categories
    def _query_database_multiple_categories(self, image_embedding, categories, k):
        candidate_items_dict = {}
        for category in categories:
            candidate_items_dict[category] = self._query_database_one_category(image_embedding, category, k)
        return candidate_items_dict
    
    def _generate_combinations(self, candidate_items_dict):
        combos = list(product(*candidate_items_dict.values()))
        return combos
    
    def _get_best_combo(self, image_embedding, combos):
        best_score = 0
        best_combo = None
        for combo in combos:
            embeddings = [embedding for _, _, embedding in combo]
            embeddings.append(image_embedding)
            score = self.scoring.score(embeddings)
            if score > best_score:
                best_score = score
                best_combo = combo
        return best_combo