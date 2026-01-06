# -*- coding: utf-8 -*-
"""True_Third_party_NLP_Project_py.ipynb


Original file is located at
    https://colab.research.google.com/drive/1IimyezKA4rxYUc7u2JlZVSIMVBjR5joy
"""
# AI Third Party Risk Management project questionnaire for IT 745
# Install SteramLit

!pip install --upgrade scikit-learn
import streamlit as st
import pandas as pd
import json
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import PyPDF2

# # =========================================================
# # Questionnaire Configuration
# # =========================================================
# questions = {
#     "q1": {"text": "Does the vendor have a formal cybersecurity policy?", "weight": 10},
#     "q2": {"text": "Is data encrypted in transit and at rest?", "weight": 10},
#     "q3": {"text": "Does the vendor perform regular vulnerability assessments?", "weight": 10},
#     "q4": {"text": "Is multi-factor authentication (MFA) implemented?", "weight": 8},
#     "q5": {"text": "Does the vendor comply with relevant regulations (e.g., GDPR, HIPAA)?", "weight": 10},
#     "q6": {"text": "Does the vendor store or process sensitive data?", "weight": 7},
#     "q7": {"text": "Does the vendor subcontract any critical services?", "weight": 8},
#     "q8": {"text": "Is the vendor’s incident response plan tested annually?", "weight": 10},
#     "q9": {"text": "Has the vendor experienced any recent data breaches?", "weight": 12},
#     "q10": {"text": "Does the vendor provide employee cybersecurity training?", "weight": 5}
# }
# 
# # =========================================================
# # Control Corpus (Baseline for NLP)
# # =========================================================
# CONTROL_CORPUS = [
#     "access control policy least privilege authentication authorization",
#     "incident response detection containment eradication recovery testing",
#     "data protection encryption classification retention privacy",
#     "business continuity disaster recovery testing resilience",
#     "vendor risk governance oversight compliance monitoring"
# ]
# 
# # =========================================================
# # Helper Functions
# # =========================================================
# def extract_policy_text(uploaded_file):
#     text = ""
#     if uploaded_file.type == "application/pdf":
#         reader = PyPDF2.PdfReader(uploaded_file)
#         for page in reader.pages:
#             text += page.extract_text() or ""
#     elif uploaded_file.type == (
#         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#     ):
#         doc = docx.Document(uploaded_file)
#         for para in doc.paragraphs:
#             text += para.text + " "
#     return text.lower()
# 
# 
# def calculate_similarity(policy_text):
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf = vectorizer.fit_transform([policy_text] + CONTROL_CORPUS)
#     scores = cosine_similarity(tfidf[0:1], tfidf[1:])
#     return float(np.mean(scores))
# 
# 
# def calculate_risk_score(responses, criticality):
#     total_weight = sum(q["weight"] for q in questions.values())
#     weighted_score = 0
# 
#     for key, response in responses.items():
#         weight = questions[key]["weight"]
#         if response == "Yes":
#             weighted_score += weight
#         elif response == "Partial":
#             weighted_score += weight * 0.5
# 
#     score = (weighted_score / total_weight) * 100
# 
#     if criticality == "High":
#         score *= 0.9
#     elif criticality == "Low":
#         score *= 1.1
# 
#     return min(100, max(0, score))
# 
# 
# def classify_risk(score):
#     if score < 50:
#         return "High Risk"
#     elif score < 75:
#         return "Medium Risk"
#     else:
#         return "Low Risk"
# 
# 
# def generate_recommendations(responses, risk_class):
#     recs = []
# 
#     if risk_class == "High Risk":
#         recs.extend([
#             "Conduct an immediate security review and request remediation evidence.",
#             "Perform an audit focusing on encryption, access control, and incident response.",
#             "Require formal SLAs for breach notification and monitoring."
#         ])
#     elif risk_class == "Medium Risk":
#         recs.extend([
#             "Request updated compliance reports (SOC 2, ISO 27001).",
#             "Increase MFA coverage and incident response testing."
#         ])
#     else:
#         recs.extend([
#             "Maintain annual assessments and continuous monitoring.",
#             "Sustain communication with vendor security leadership."
#         ])
# 
#     if responses.get("q9") == "Yes":
#         recs.append("Recent breach identified — verify corrective actions.")
#     if responses.get("q7") == "Yes":
#         recs.append("Subcontractors used — ensure downstream security equivalence.")
#     if responses.get("q10") == "No":
#         recs.append("Employee training gap — mandate annual cybersecurity training.")
# 
#     return recs
# 
# 
# def llm_policy_analysis(policy_text, similarity_score):
#     system_prompt = (
#         "You are a cybersecurity risk analyst. "
#         "Provide descriptive analysis only. "
#         "Do not assign risk levels or make decisions."
#     )
# 
#     user_prompt = f"""
# NLP similarity score (0–1): {similarity_score:.2f}
# 
# Analyze the vendor policy and provide:
# 1. Observed strengths
# 2. Identified gaps
# 3. Suggested improvements
# 
# Policy text:
# {policy_text[:6000]}
# """
# 
#     response = client.chat.completions.create(
#         model=DEPLOYMENT_NAME,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=400
#     )
# 
#     return response.choices[0].message.content
# 
# # =========================================================
# # Streamlit UI
# # =========================================================
# st.set_page_config(
#     page_title="AI-Driven Third-Party Risk Assessment",
#     layout="centered"
# )
# 
# st.title("🧠 AI-Driven Third-Party Risk Assessment Dashboard")
# st.write("Assess vendor cybersecurity readiness using questionnaires, NLP, and AI-assisted analysis.")
# 
# # Vendor Information
# st.header("Vendor Information")
# vendor = st.text_input("Vendor Name", "SampleVendorCo")
# industry = st.text_input("Vendor Industry", "Finance")
# criticality = st.selectbox("Vendor Criticality", ["High", "Medium", "Low"])
# 
# # Questionnaire
# st.header("Risk Assessment Questionnaire")
# responses = {}
# for key, q in questions.items():
#     responses[key] = st.radio(q["text"], ["Yes", "No", "Partial"], horizontal=True)
# 
# # Policy Upload
# st.header("Upload Vendor Policy Documentation")
# uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
# 
# # Run Assessment
# if st.button("Run AI-Driven Assessment"):
# 
#     score = calculate_risk_score(responses, criticality)
#     risk_class = classify_risk(score)
#     recommendations = generate_recommendations(responses, risk_class)
#     timestamp = datetime.utcnow().isoformat() + "Z"
# 
#     similarity_score = None
#     llm_output = None
# 
#     if uploaded_file:
#         policy_text = extract_policy_text(uploaded_file)
#         similarity_score = calculate_similarity(policy_text)
# 
#         with st.spinner("Running Azure OpenAI analysis..."):
#             llm_output = llm_policy_analysis(policy_text, similarity_score)
# 
#     # Results
#     st.subheader("Assessment Results")
#     st.metric("Risk Score", f"{round(score,1)} / 100")
#     st.metric("Risk Classification", risk_class)
#     st.progress(int(score))
# 
#     if similarity_score is not None:
#         st.metric("NLP Similarity Score", f"{similarity_score:.2f}")
# 
#     st.write("### 🧩 AI-Generated Recommendations")
#     for r in recommendations:
#         st.markdown(f"- {r}")
# 
#     if llm_output:
#         st.write("### 🤖 LLM-Assisted Policy Analysis")
#         st.write(llm_output)
# 
#     # Export
#     result = {
#         "vendor": vendor,
#         "industry": industry,
#         "criticality": criticality,
#         "risk_score": round(score, 1),
#         "risk_classification": risk_class,
#         "nlp_similarity_score": similarity_score,
#         "recommendations": recommendations,
#         "llm_analysis": llm_output,
#         "responses": responses,
#         "timestamp": timestamp
#     }
# 
#     st.download_button(
#         "Download JSON Report",
#         json.dumps(result, indent=4),
#         f"AI_TPRM_Report_{vendor}.json",
#         "application/json"
#     )
# 
#     df = pd.DataFrame([result])
#     st.download_button(
#         "Download CSV Summary",
#         df.to_csv(index=False),
#         f"AI_TPRM_Report_{vendor}.csv",
#         "text/csv"
#     )
# 
# st.markdown("---")
# st.caption("AI-Driven Cybersecurity Research on Third-Party Risk Management © 2026")
