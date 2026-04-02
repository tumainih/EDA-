def generate_recommendations(feature_importances_df, task_type):
    """
    Generates dynamic, localized recommendations based on feature importances
    and the type of problem being solved.
    """
    if feature_importances_df is None or feature_importances_df.empty:
        return ["Not enough data to formulate recommendations."]

    recommendations = []
    
    top_feature = feature_importances_df.iloc[0]['Feature']
    second_feature = feature_importances_df.iloc[1]['Feature'] if len(feature_importances_df) > 1 else None

    # Foundational Rule: High importance means high impact.
    recommendations.append(
        f"**Focus Area 1**: The feature '{top_feature}' has the highest predictive power for your target. "
        f"Monitoring and optimizing '{top_feature}' should be your immediate priority."
    )

    if second_feature:
        recommendations.append(
            f"**Focus Area 2**: Similarly, '{second_feature}' holds significant weight. Investigating the interplay "
            f"between '{top_feature}' and '{second_feature}' could yield compounding improvements."
        )

    if task_type.lower() == 'classification':
        recommendations.append(
            "**Classification Strategy**: Since you are predicting discrete categories, ensure that the classes are balanced. "
            "If your accuracy is high but practical results vary, consider checking for class imbalances in your dataset."
        )
    else:
        recommendations.append(
            "**Regression Strategy**: You are predicting a continuous outcome. If the R-Squared value is lower than desired, "
            "it indicates that the current variables do not fully explain the variance. You may need to collect additional "
            "external features (like seasonality, market trends) to improve prediction accuracy."
        )

    return recommendations
