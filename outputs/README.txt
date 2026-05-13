# Generated output files written here by the notebook:
#
# outputs/figures/
#   class_distribution.png       — Target class balance bar chart
#   correlation_heatmap.png      — Lower-triangle feature correlation heatmap
#   feature_distributions.png    — Key features split by diagnosis class
#   confusion_matrices.png       — Side-by-side confusion matrices (RF + LR)
#   roc_comparison.png           — Overlaid ROC curves with AUC scores
#   rf_feature_importance.png    — Top 20 RF Gini importances
#   lr_coefficients.png          — LR coefficient polarity chart
#   metric_comparison.png        — Grouped bar chart of Accuracy / F1 / AUC
#
# outputs/models/
#   random_forest.pkl            — Best fitted RandomForestClassifier
#   logistic_regression.pkl      — Best fitted LogisticRegression
#   scaler.pkl                   — Fitted StandardScaler (required for inference)
#
# outputs/
#   results_summary.csv          — Head-to-head metric comparison table
#
# Run the notebook end-to-end to regenerate all outputs.
# To reload a saved model:
#   import joblib
#   model  = joblib.load('outputs/models/random_forest.pkl')
#   scaler = joblib.load('outputs/models/scaler.pkl')
