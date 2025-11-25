"""
Churn prediction inference module.

This module loads the trained XGBoost model and preprocessing pipeline
to make predictions on customer data.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import sys
from sklearn.base import BaseEstimator, TransformerMixin

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_PATH, PIPELINE_PATH
from database.queries import ChurnDatabase


# ==================== CUSTOM PREPROCESSOR ====================
# This class is required for unpickling the preprocessing pipeline

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor for customer churn data.
    Handles data type conversions and boolean mapping.
    """
    def __init__(self):
        self.boolean_columns = None
        
    def fit(self, X, y=None):
        # Identify boolean columns (columns with exactly 2 unique values)
        self.boolean_columns = [col for col in X.columns if X[col].nunique() == 2]
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Convert TotalCharges to numeric
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        
        # 2. Convert SeniorCitizen to object type
        if 'SeniorCitizen' in X.columns:
            X['SeniorCitizen'] = X['SeniorCitizen'].astype('object')
        
        # 3. Map boolean columns (Yes/No, Female/Male) to 0/1
        for col in self.boolean_columns:
            if col in X.columns:
                unique_vals = set(X[col].dropna().unique())
                if unique_vals.issubset({'Yes', 'No'}):
                    X[col] = X[col].map({'Yes': 1, 'No': 0})
                elif unique_vals.issubset({'Female', 'Male'}):
                    X[col] = X[col].map({'Female': 0, 'Male': 1})
        
        # 4. Drop customerID if exists
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])
        
        return X


class ChurnPredictor:
    """
    Singleton class for churn prediction.
    Loads model and pipeline once for efficient inference.
    """
    
    _instance = None
    _model = None
    _pipeline = None
    _model_loaded = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(ChurnPredictor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize predictor and load model/pipeline if not already loaded."""
        if not ChurnPredictor._model_loaded:
            self._load_model_and_pipeline()
            ChurnPredictor._model_loaded = True
    
    def _load_model_and_pipeline(self):
        """Load the trained model and preprocessing pipeline."""
        try:
            # Custom unpickler to handle CustomPreprocessor from __main__
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Redirect __main__.CustomPreprocessor to our module
                    if module == '__main__' and name == 'CustomPreprocessor':
                        return CustomPreprocessor
                    return super().find_class(module, name)
            
            # Load preprocessing pipeline with custom unpickler
            with open(PIPELINE_PATH, 'rb') as f:
                ChurnPredictor._pipeline = CustomUnpickler(f).load()
            print(f"✅ Pipeline loaded from {PIPELINE_PATH}")
            
            # Load XGBoost model
            with open(MODEL_PATH, 'rb') as f:
                ChurnPredictor._model = pickle.load(f)
            print(f"✅ Model loaded from {MODEL_PATH}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model or pipeline file not found. Please ensure files exist:\n"
                f"  - Model: {MODEL_PATH}\n"
                f"  - Pipeline: {PIPELINE_PATH}\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading model or pipeline: {e}")
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on churn probability.
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Risk level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        if probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_confidence(self, probability: float) -> str:
        """
        Determine prediction confidence based on probability.
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Confidence level: 'high' or 'medium'
        """
        # High confidence when probability is very high or very low
        if probability > 0.8 or probability < 0.2:
            return 'high'
        else:
            return 'medium'
    
    def predict_single_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Predict churn probability for a single customer by ID.
        
        Args:
            customer_id: Customer ID to predict
            
        Returns:
            Dictionary with prediction results:
            - customer_id: The customer ID
            - churn_probability: Probability of churn (0-1)
            - churn_probability_pct: Formatted percentage string
            - risk_level: 'LOW', 'MEDIUM', or 'HIGH'
            - confidence: 'high' or 'medium'
            - prediction: 'CHURN' or 'RETAIN'
            - error: Error message if prediction failed
        """
        try:
            # Fetch customer data from database
            db = ChurnDatabase()
            customer_data = db.get_customer_by_id(customer_id)
            
            if customer_data.empty:
                return {
                    'customer_id': customer_id,
                    'error': f"Customer {customer_id} not found in database"
                }
            
            # Remove target variable if it exists
            X = customer_data.drop(columns=['Churn'], errors='ignore')
            
            # Transform using preprocessing pipeline
            X_transformed = self._pipeline.transform(X)
            
            # Get prediction probability
            # predict_proba returns [[prob_no_churn, prob_churn]]
            proba = self._model.predict_proba(X_transformed)[0]
            churn_prob = float(proba[1])  # Probability of churn
            
            # Get binary prediction (0.5 threshold)
            prediction = 'CHURN' if churn_prob >= 0.5 else 'RETAIN'
            
            return {
                'customer_id': customer_id,
                'churn_probability': churn_prob,
                'churn_probability_pct': f"{churn_prob * 100:.1f}%",
                'risk_level': self._get_risk_level(churn_prob),
                'confidence': self._get_confidence(churn_prob),
                'prediction': prediction,
                'customer_data': customer_data.iloc[0].to_dict()  # Include raw data
            }
            
        except Exception as e:
            return {
                'customer_id': customer_id,
                'error': f"Prediction failed: {str(e)}"
            }
    
    def predict_batch(self, customer_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Predict churn for multiple customers (batch processing).
        
        Args:
            customer_ids: List of customer IDs
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        try:
            # Fetch all customers at once (more efficient)
            db = ChurnDatabase()
            all_customers = db.get_all_customers()
            
            # Filter to requested IDs
            customers = all_customers[all_customers['customerID'].isin(customer_ids)]
            
            if customers.empty:
                return [{
                    'error': 'No customers found with provided IDs',
                    'customer_ids': customer_ids
                }]
            
            # Store customer IDs before transformation
            customer_id_list = customers['customerID'].tolist()
            
            # Remove target variable and ID
            X = customers.drop(columns=['Churn', 'customerID'], errors='ignore')
            
            # Transform and predict
            X_transformed = self._pipeline.transform(X)
            probabilities = self._model.predict_proba(X_transformed)[:, 1]
            
            # Build results
            for cid, prob in zip(customer_id_list, probabilities):
                prob = float(prob)
                results.append({
                    'customer_id': cid,
                    'churn_probability': prob,
                    'churn_probability_pct': f"{prob * 100:.1f}%",
                    'risk_level': self._get_risk_level(prob),
                    'prediction': 'CHURN' if prob >= 0.5 else 'RETAIN'
                })
            
            return results
            
        except Exception as e:
            return [{
                'error': f"Batch prediction failed: {str(e)}",
                'customer_ids': customer_ids
            }]
    
    def predict_from_raw_data(self, raw_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Predict churn from raw customer data (for what-if analysis).
        
        Args:
            raw_data: Dictionary or DataFrame with customer features
                     Must contain all required fields in same format as training data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(raw_data, dict):
                df = pd.DataFrame([raw_data])
            else:
                df = raw_data.copy()
            
            # Remove target variable if exists
            if 'Churn' in df.columns:
                df = df.drop(columns=['Churn'])
            
            # Remove customerID if exists (not a feature)
            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])
            
            # Transform using pipeline
            X_transformed = self._pipeline.transform(df)
            
            # Predict
            proba = self._model.predict_proba(X_transformed)[0]
            churn_prob = float(proba[1])
            
            return {
                'churn_probability': churn_prob,
                'churn_probability_pct': f"{churn_prob * 100:.1f}%",
                'risk_level': self._get_risk_level(churn_prob),
                'confidence': self._get_confidence(churn_prob),
                'prediction': 'CHURN' if churn_prob >= 0.5 else 'RETAIN',
                'input_data': raw_data if isinstance(raw_data, dict) else raw_data.iloc[0].to_dict()
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'input_data': raw_data if isinstance(raw_data, dict) else None
            }
    
    def get_high_risk_customers(
        self, 
        threshold: float = 0.7, 
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """
        Find customers with high churn probability.
        
        Args:
            threshold: Minimum churn probability (default: 0.7)
            limit: Maximum number of customers to return
            
        Returns:
            List of high-risk customers sorted by churn probability (descending)
        """
        try:
            # Get all customers
            db = ChurnDatabase()
            all_customers = db.get_all_customers()
            
            # Store customer IDs
            customer_ids = all_customers['customerID'].tolist()
            
            # Remove target and ID for prediction
            X = all_customers.drop(columns=['Churn', 'customerID'], errors='ignore')
            
            # Transform and predict
            X_transformed = self._pipeline.transform(X)
            probabilities = self._model.predict_proba(X_transformed)[:, 1]
            
            # Build results with probabilities
            results = []
            for cid, prob in zip(customer_ids, probabilities):
                if prob >= threshold:
                    results.append({
                        'customer_id': cid,
                        'churn_probability': float(prob),
                        'churn_probability_pct': f"{prob * 100:.1f}%",
                        'risk_level': self._get_risk_level(prob)
                    })
            
            # Sort by probability (descending)
            results.sort(key=lambda x: x['churn_probability'], reverse=True)
            
            # Apply limit
            if limit:
                results = results[:limit]
            
            return results
            
        except Exception as e:
            return [{
                'error': f"Failed to get high-risk customers: {str(e)}"
            }]
    
    def predict_all_customers(self) -> pd.DataFrame:
        """
        Run predictions on all customers in database.
        
        Returns:
            DataFrame with customer_id, churn_probability, risk_level
        """
        try:
            # Get all customers
            db = ChurnDatabase()
            all_customers = db.get_all_customers()
            
            # Store customer IDs
            customer_ids = all_customers['customerID'].tolist()
            
            # Remove target and ID
            X = all_customers.drop(columns=['Churn', 'customerID'], errors='ignore')
            
            # Transform and predict
            X_transformed = self._pipeline.transform(X)
            probabilities = self._model.predict_proba(X_transformed)[:, 1]
            
            # Build results DataFrame
            results_df = pd.DataFrame({
                'customer_id': customer_ids,
                'churn_probability': probabilities,
                'risk_level': [self._get_risk_level(p) for p in probabilities],
                'prediction': ['CHURN' if p >= 0.5 else 'RETAIN' for p in probabilities]
            })
            
            return results_df
            
        except Exception as e:
            print(f"Error predicting all customers: {e}")
            return pd.DataFrame()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        try:
            return {
                'model_type': type(self._model).__name__,
                'model_path': MODEL_PATH,
                'pipeline_path': PIPELINE_PATH,
                'model_params': self._model.get_params() if hasattr(self._model, 'get_params') else {},
                'loaded': self._model_loaded
            }
        except Exception as e:
            return {'error': str(e)}


# ==================== CONVENIENCE FUNCTIONS ====================

def predict_customer(customer_id: str) -> Dict[str, Any]:
    """Quick function to predict churn for a customer."""
    predictor = ChurnPredictor()
    return predictor.predict_single_customer(customer_id)


def get_high_risk_customers(threshold: float = 0.7, limit: int = 50) -> List[Dict[str, Any]]:
    """Quick function to get high-risk customers."""
    predictor = ChurnPredictor()
    return predictor.get_high_risk_customers(threshold, limit)


def predict_from_data(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Quick function to predict from raw data."""
    predictor = ChurnPredictor()
    return predictor.predict_from_raw_data(customer_data)
