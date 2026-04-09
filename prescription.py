def generate_recommendations(feature_importances_df, task_type):
    """
    Generates strategic, reasoning-based recommendations.
    """
    if feature_importances_df is None or feature_importances_df.empty:
        return ["Insufficient data for reasoning."]

    top_feature = feature_importances_df.iloc[0]['Feature']
    top_score = feature_importances_df.iloc[0]['Importance']
    
    recs = [
        f"### 🎯 Strategic Priority: {top_feature}",
        f"The engine has identified **{top_feature}** as your most critical lever, with an importance score of **{top_score:.2f}**. This means changes in this variable will have the highest impact on your final outcomes.",
        "### 🧠 Reasoning",
        f"Based on historical patterns, your target outcome is sensitive to {top_feature}. We recommend focusing operational improvements here to maximize efficiency."
    ]
    
    if len(feature_importances_df) > 1:
        v2 = feature_importances_df.iloc[1]['Feature']
        recs.append(f"### 🤝 Secondary Synergy: {v2}")
        recs.append(f"Monitor the correlation between {top_feature} and {v2} to identify compounding risks or opportunities.")

    return "\n\n".join(recs)

def get_bot_response(query, context):
    """
    Simulates a data expert response based on query and dataset context.
    """
    query = query.lower()
    if "what-if" in query or "scenario" in query:
        return "You can use the 'What-If Explorer' in the Predictive tab to simulate changes in real-time."
    elif "important" in query or "driver" in query:
        return f"The primary driver of your success is {context.get('top_feature', 'unknown')}. Focus your resources there."
    elif "missing" in query or "quality" in query:
        return f"The dataset had {context.get('missing_total', 0)} missing values which we've imputed to ensure stability."
    else:
        return "I'm the Nexus Intelligence Bot. I can help you understand the causal factors and predictions in your data. Try asking about 'top drivers' or 'data quality'."
