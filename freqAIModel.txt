import logging
from typing import Any, Dict, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMRegressor as LGBM
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class LightGBMRegressor(BaseRegressionModel):
    """
    LightGBM-basiertes Regressionsmodell für FreqAI mit verbessertem Logging und Fehlerbehandlung
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = None
        logger.info("LightGBM Regressor initialisiert")

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Verbessertes Model-Training mit detailliertem Logging
        """
        try:
            X = data_dictionary["train_features"]
            y = data_dictionary["train_labels"]
            
            # Erweiterte Logging-Informationen
            logger.info(f"""
            Starte Model-Training:
            - Anzahl Samples: {len(X)}
            - Anzahl Features: {X.shape[1]}
            - Feature Namen: {', '.join(dk.training_features_list)}
            - Label Verteilung: Positiv={sum(y > 0)}, Negativ={sum(y <= 0)}
            """)

            # Model Parameter Logging
            model_params = {
                "n_estimators": self.model_training_parameters.get("n_estimators", 800),
                "learning_rate": self.model_training_parameters.get("learning_rate", 0.01),
                "max_depth": self.model_training_parameters.get("max_depth", 5),
                "num_leaves": self.model_training_parameters.get("num_leaves", 25),
                "min_data_in_leaf": self.model_training_parameters.get("min_data_in_leaf", 15),
                "feature_fraction": self.model_training_parameters.get("feature_fraction", 0.8),
                "bagging_fraction": self.model_training_parameters.get("bagging_fraction", 0.8),
                "bagging_freq": self.model_training_parameters.get("bagging_freq", 3),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
            
            logger.info(f"Model Parameter: {model_params}")

            model = LGBM(**model_params)
            model.fit(X, y)
            self.model = model

            # Feature Importance Analyse
            feature_imp = pd.DataFrame({
                'feature': dk.training_features_list,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"""
            Training abgeschlossen:
            Top 5 wichtigste Features:
            {feature_imp.head().to_string()}
            """)

            return self.model

        except Exception as e:
            logger.error(f"""
            Fehler während des Trainings:
            - Error: {str(e)}
            - Daten Shape: X={X.shape if 'X' in locals() else 'N/A'}, y={y.shape if 'y' in locals() else 'N/A'}
            """)
            raise e

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], pd.DataFrame]:
        """
        Verbesserte Vorhersagen mit Konfidenz-Scoring
        """
        try:
            # Feature Extraktion
            dk.find_features(unfiltered_df)
            filtered_df, _ = dk.filter_features(
                unfiltered_df, dk.training_features_list, training_filter=False
            )

            predictions = self.model.predict(filtered_df)
            
            # Verbesserte Konfidenz-Berechnung
            do_predict = np.ones(len(predictions))
            
            # Konfidenz-basierte Filterung
            confidence_threshold = 0.65  # Anpassbar
            if hasattr(self.model, 'predict_proba'):
                predictions_proba = self.model.predict_proba(filtered_df)
                # Setze do_predict auf 0 für niedrige Konfidenz
                max_proba = np.max(predictions_proba, axis=1)
                do_predict[max_proba < confidence_threshold] = 0
            else:
                predictions_proba = np.zeros((len(predictions), 2))
                
            logger.info(f"""
            Prediction Stats:
            - Samples: {len(predictions)}
            - Durchschnitt: {predictions.mean():.4f}
            - Std: {predictions.std():.4f}
            - Konfidente Signale: {sum(do_predict)}/{len(do_predict)}
            """)

            return predictions, do_predict, predictions_proba

        except Exception as e:
            logger.error(f"Vorhersage-Fehler: {str(e)}")
            raise e