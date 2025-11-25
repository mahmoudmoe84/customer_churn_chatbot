# Test Queries for Churn Prediction Chatbot

Use these queries in the Streamlit app to test the model with unseen data.

## 🔴 Test Case 1: High Risk Customer (Expected: HIGH)
**Scenario:** Young customer, 3 months tenure, fiber optic, no security services, month-to-month

**Query:**
```
Predict churn for a hypothetical customer with these characteristics:
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 268.5
}
```

---

## 🟢 Test Case 2: Low Risk Customer (Expected: LOW)
**Scenario:** Senior with 68 months tenure, two-year contract, comprehensive security

**Query:**
```
What's the churn probability for this customer:
{
  "gender": "Male",
  "SeniorCitizen": 1,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 68,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 54.75,
  "TotalCharges": 3722.5
}
```

---

## 🟡 Test Case 3: Medium Risk Customer (Expected: MEDIUM)
**Scenario:** 12 months tenure, no internet, month-to-month contract

**Query:**
```
Analyze churn risk for:
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "No",
  "OnlineSecurity": "No internet service",
  "OnlineBackup": "No internet service",
  "DeviceProtection": "No internet service",
  "TechSupport": "No internet service",
  "StreamingTV": "No internet service",
  "StreamingMovies": "No internet service",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Mailed check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 358.2
}
```

---

## 🔴 Test Case 4: Very High Risk (Expected: HIGH)
**Scenario:** Brand new customer (1 month), fiber optic, electronic check, no add-ons

**Query:**
```
What are the chances this new customer will churn?
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 70.0
}
```

---

## 🟢 Test Case 5: Very Low Risk (Expected: LOW)
**Scenario:** 72 months tenure, senior with family, two-year contract, all services

**Query:**
```
Predict churn for this loyal customer:
{
  "gender": "Female",
  "SeniorCitizen": 1,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 72,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Credit card (automatic)",
  "MonthlyCharges": 103.7,
  "TotalCharges": 7466.4
}
```

---

## 🟡 Test Case 6: Edge Case - No Phone Service
**Scenario:** 24 months tenure, no phone but has fiber optic internet with security

**Query:**
```
Can you predict churn for this customer who doesn't have phone service?
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 75.25,
  "TotalCharges": 1806.0
}
```

---

## 📝 Quick Test Queries (Natural Language)

You can also test with simpler queries:

1. **"What factors most influence customer churn?"**
2. **"Show me churn rate for customers without landline"** *(Your special requirement!)*
3. **"Find 5 customers with highest churn risk"**
4. **"What's the overall churn rate?"**
5. **"Analyze churn for senior citizens"**
6. **"Compare churn rates between contract types"**

---

## 🎯 Expected Model Behavior

The model should demonstrate:

- **High accuracy** on clear high-risk indicators (short tenure + month-to-month + fiber optic + no security)
- **Low predictions** for loyal customers (long tenure + contract commitment + comprehensive services)
- **Nuanced predictions** for edge cases (no phone service, senior citizens, etc.)
- **Confidence levels** that match the scenario complexity

---

## 📊 Validation Checklist

After testing, verify:

- [ ] High-risk scenarios get >70% churn probability
- [ ] Low-risk scenarios get <30% churn probability
- [ ] Medium-risk falls between 40-70%
- [ ] Risk levels (HIGH/MEDIUM/LOW) match expectations
- [ ] Confidence scores are reasonable
- [ ] Agent reasoning is logical and uses appropriate tools
- [ ] Formatting is clear and readable
- [ ] No errors in prediction or data handling
