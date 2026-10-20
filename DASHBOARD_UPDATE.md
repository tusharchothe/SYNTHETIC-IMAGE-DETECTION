# Dashboard Tab Added - Separate ResNet & CLIP Sections

## Overview

Added a new **Dashboard** tab to the Gradio UI with separate analytics sections for ResNet-50 and CLIP predictions.

## New Tab Structure

The UI now has **6 tabs** instead of 5:

1. **Upload & Detect** — Upload images and run inference
2. **Dashboard** ⭐ NEW — Compare ResNet vs CLIP performance
3. **Pending Review Queue** — Browse images awaiting review
4. **Review & Label** — Submit human labels
5. **Dataset Management** — Build versioned datasets
6. **Training & Models** — Run training and view runs

## Dashboard Features

### 🔵 ResNet-50 Section

Shows ResNet-50 model predictions:
- **Total Predictions** — Number of images processed
- **Predicted Real** — Count of real classifications
- **Predicted AI** — Count of AI-generated classifications
- **Average Confidence Score** — Mean raw score across predictions
- **Real %** — Percentage predicted as real
- **AI %** — Percentage predicted as AI

### 🟢 CLIP Section

Shows CLIP model predictions with identical metrics:
- **Total Predictions**
- **Predicted Real**
- **Predicted AI**
- **Average Confidence Score**
- **Real %**
- **AI %**

### 🔄 Model Agreement Analysis

Cross-model comparison metrics:
- **Total Images** — Overall image count
- **Models Agree** — Count where both models predict same class
- **Models Disagree** — Count where models predict different classes
- **Agreement Rate** — Percentage of agreement

## Usage

1. Go to the **Dashboard** tab
2. Click **"Refresh Statistics"**
3. View real-time analytics for both models
4. Compare performance side-by-side
5. Identify disagreement cases for human review

## What It Shows

### ResNet-50 Example Output
```
Total Predictions: 150
Predicted Real: 85
Predicted AI: 65
Average Confidence Score: 0.8742
Real %: 56.7%
AI %: 43.3%
```

### CLIP Example Output
```
Total Predictions: 150
Predicted Real: 92
Predicted AI: 58
Average Confidence Score: 0.8156
Real %: 61.3%
AI %: 38.7%
```

### Agreement Analysis
```
Total Images: 150
Models Agree: 132
Models Disagree: 18
Agreement Rate: 88.0%
```

## Use Cases

### 1. Model Performance Comparison
Quickly see which model is more conservative or aggressive in predictions.

### 2. Quality Assurance
High disagreement rates indicate images that need human review.

### 3. Dataset Balance Monitoring
Track real vs AI distribution before building datasets.

### 4. Confidence Analysis
Average scores help identify model certainty levels.

### 5. Disagreement Triage
Find images where models disagree → prioritize for review.

## Implementation Details

### Data Source
- Fetches from `/queue` endpoint (pending images)
- Processes predictions client-side in Gradio
- No database changes required

### Statistics Calculated
```python
stats = {
    "total_images": len(items),
    "resnet": {"real": 0, "ai": 0, "total": 0, "avg_score": 0.0},
    "clip": {"real": 0, "ai": 0, "total": 0, "avg_score": 0.0},
    "agreement": 0,
    "disagreement": 0,
}
```

### Agreement Logic
- **Agree**: Both models predict same class (real or ai)
- **Disagree**: Models predict different classes
- Only counts images with predictions from both models

## Visual Layout

```
┌─────────────────────────────────────────────────┐
│           Model Performance Dashboard            │
│  Compare ResNet-50 and CLIP predictions         │
├─────────────────────────────────────────────────┤
│  [Refresh Statistics]                           │
│                                                  │
│  ┌──────────────────┬──────────────────┐        │
│  │  🔵 ResNet-50    │  🟢 CLIP         │        │
│  ├──────────────────┼──────────────────┤        │
│  │ Total: 150       │ Total: 150       │        │
│  │ Real: 85         │ Real: 92         │        │
│  │ AI: 65           │ AI: 58           │        │
│  │ Avg Score: 0.874 │ Avg Score: 0.816 │        │
│  │ Real%: 56.7%     │ Real%: 61.3%     │        │
│  │ AI%: 43.3%       │ AI%: 38.7%       │        │
│  └──────────────────┴──────────────────┘        │
│                                                  │
│  🔄 Model Agreement Analysis                     │
│  ┌──────────────────────────────────────┐       │
│  │ Total Images: 150                    │       │
│  │ Models Agree: 132                    │       │
│  │ Models Disagree: 18                  │       │
│  │ Agreement Rate: 88.0%                │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

## Benefits

✅ **Side-by-Side Comparison** — Easy visual comparison of both models  
✅ **Real-Time Stats** — Refresh anytime to see current state  
✅ **Agreement Tracking** — Identify cases needing human review  
✅ **Performance Monitoring** — Track model behavior over time  
✅ **No Database Changes** — Uses existing queue endpoint  

## Workflow Integration

### Before Training
1. Upload images → check Dashboard
2. Review disagreement cases first
3. Build balanced dataset

### After Training
1. Upload test images
2. Compare new vs old model predictions
3. Verify improvements in Dashboard

### Quality Control
1. Refresh Dashboard regularly
2. High disagreement? → More reviews needed
3. Track prediction distribution

## Technical Notes

- **Refresh is Manual** — Click button to update stats
- **Limit: 1000 Images** — Fetches up to 1000 from queue for stats
- **Client-Side Processing** — All calculations in browser
- **No Persistence** — Stats computed on-the-fly each refresh

## Example Scenarios

### Scenario 1: Balanced Dataset
```
ResNet: 50% real, 50% ai
CLIP: 48% real, 52% ai
Agreement: 92%
→ Good balance, high agreement
```

### Scenario 2: Model Bias
```
ResNet: 70% real, 30% ai
CLIP: 45% real, 55% ai
Agreement: 65%
→ ResNet may be biased toward real, low agreement
```

### Scenario 3: Uncertain Data
```
ResNet: 55% real, 45% ai
CLIP: 53% real, 47% ai
Agreement: 60%
→ Many disagreements, more reviews needed
```

## Future Enhancements

Potential additions:
- 📊 Charts/graphs for visual comparison
- 📈 Historical trends over time
- 🎯 Precision/recall metrics (requires human labels)
- 🔍 Filter by date range
- 📁 Export statistics to CSV
- 🎨 Color-coded agreement heatmap

## Summary

The new Dashboard tab provides:
- **Separate sections** for ResNet-50 and CLIP
- **Performance metrics** for each model
- **Agreement analysis** between models
- **Easy comparison** side-by-side

Perfect for monitoring model behavior and identifying images that need human review!
