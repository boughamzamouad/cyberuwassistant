import json
import time
from typing import Dict, Any, List
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from openai import OpenAI, APIError, RateLimitError

# Optional PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("ReportLab not installed. PDF download disabled. Run: pip install reportlab")

# Secure loading of API key and all sensitive prompts
try:
    XAI_API_KEY = st.secrets["XAI_API_KEY"]
    SYSTEM_PROMPT = st.secrets["prompts"]["SYSTEM_PROMPT"]
    INCIDENTS_PROMPT_TEMPLATE = st.secrets["prompts"]["INCIDENTS_PROMPT_TEMPLATE"]
    EXPOSURES_PROMPT_TEMPLATE = st.secrets["prompts"]["EXPOSURES_PROMPT_TEMPLATE"]
    SIMILAR_EVENTS_PROMPT_TEMPLATE = st.secrets["prompts"]["SIMILAR_EVENTS_PROMPT_TEMPLATE"]
    VULNERABILITY_PROMPT_TEMPLATE = st.secrets["prompts"]["VULNERABILITY_PROMPT_TEMPLATE"]
    TARGETED_GANGS_PROMPT_TEMPLATE = st.secrets["prompts"]["TARGETED_GANGS_PROMPT_TEMPLATE"]
    LOSS_ESTIMATION_PROMPT_TEMPLATE = st.secrets["prompts"]["LOSS_ESTIMATION_PROMPT_TEMPLATE"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets.toml correctly.")
    st.stop()

client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY, timeout=90)

MODEL = "grok-4"

CYBER_CATEGORIES = [
    "Ransomware Susceptibility",
    "Business Email Compromise",
    "Data Breach / Privacy Liability",
    "Supply Chain / Third-Party Risk",
    "Regulatory / Compliance Exposure",
    "Business Interruption Potential",
]

TEO_CATEGORIES = [
    "Software/Product Failure Liability",
    "Intellectual Property Infringement",
    "Contractual Performance Risk",
    "Professional Negligence (Services)",
    "Privacy Regulatory Exposure",
    "Third-Party Integration Failures",
]

def research_with_grok(prompt: str, max_retries: int = 3, max_tokens: int = 1000) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except (APIError, RateLimitError) as e:
            if attempt == max_retries - 1:
                return {"error": f"API error: {str(e)}"}
            time.sleep(8 * (attempt + 1))
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"error": "Invalid JSON response"}
            time.sleep(5)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Unexpected error: {str(e)}"}
            time.sleep(10)
    return {"error": "Failed after retries"}

# ... (generate_radar_chart and generate_loss_bar_chart unchanged)

if PDF_AVAILABLE:
    def create_pdf_report(results: Dict, assess_teo: bool, selected_sections: Dict[str, bool]) -> BytesIO:
        # ... (unchanged from previous version)
        pass
else:
    def create_pdf_report(*args, **kwargs):
        return BytesIO()

@st.cache_data(ttl=86400, show_spinner=False)
def run_full_analysis(company_name: str, address: str, revenue_input: str, employees_input: str, assess_teo: bool, selected_sections: Dict[str, bool]) -> Dict:
    results = {"company": company_name}
    
    revenue_usd = revenue_input.strip() if revenue_input.strip().lower() != "unknown" and revenue_input.strip() else "Unknown"
    
    employees = employees_input.strip() if employees_input.strip().lower() != "unknown" and employees_input.strip() else None
    if employees is None:
        emp_data = research_with_grok(f"Latest estimated number of employees for {company_name}. JSON: {{'estimated_employees': 'number or range'}}", max_tokens=600)
        employees = emp_data.get("estimated_employees", "Unknown") if "error" not in emp_data else "Unknown"
    
    results.update({"revenue_usd": revenue_usd, "employees": employees})
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    active_stages = ["Gathering company basics"]
    if selected_sections["description"]: active_stages.append("Researching operations and industry")
    if selected_sections["subsidiaries"]: active_stages.append("Identifying offices/subsidiaries")
    if selected_sections["incidents"]: active_stages.append("Searching for reported incidents")
    if selected_sections["exposures"]: active_stages.append("Analyzing key cyber exposures")
    if selected_sections["similar_events"]: active_stages.append("Finding similar peer events")
    if selected_sections["vulnerability_scoring"]: active_stages.append("Scoring vulnerabilities and overview")
    if selected_sections["targeted_gangs"]: active_stages.append("Checking for targeted cyber crime gang activity")
    if selected_sections["loss_scenarios"]: active_stages.append("Estimating financial loss scenarios")
    
    stage_status = {stage: "⏳" for stage in active_stages}
    
    def update_status(current_stage: str):
        if current_stage in stage_status:
            stage_status[current_stage] = "✅"
        status_text = "\n".join(f"{status} {stage}" for stage, status in stage_status.items())
        status_placeholder.markdown(f"**Research Progress:**\n{status_text}")
        completed = sum(1 for s in stage_status.values() if s == "✅")
        progress_bar.progress(completed / len(active_stages))
    
    update_status("Gathering company basics")
    
    if selected_sections["description"]:
        update_status("Researching operations and industry")
        data = research_with_grok(f"Company: {company_name}. Brief description of operations and primary industry. Sources: official website/LinkedIn. JSON: {{'description': value}}", max_tokens=800)
        results["description"] = data.get("description", "- Not available") if "error" not in data else "- Research failed"
    
    if selected_sections["subsidiaries"]:
        update_status("Identifying offices/subsidiaries")
        data = research_with_grok(f"Company: {company_name}. List major offices and subsidiaries with locations as bullet points. JSON: {{'subsidiaries': value}}", max_tokens=600)
        results["subsidiaries"] = data.get("subsidiaries", "- None identified") if "error" not in data else "- Research failed"
    
    if selected_sections["incidents"]:
        update_status("Searching for reported incidents")
        prompt = INCIDENTS_PROMPT_TEMPLATE.format(company_name=company_name)
        data = research_with_grok(prompt, max_tokens=800)
        results["incidents"] = data.get("incidents", "- No publicly reported incidents found") if "error" not in data else "- Research failed"
    
    if selected_sections["exposures"]:
        update_status("Analyzing key cyber exposures")
        prompt = EXPOSURES_PROMPT_TEMPLATE.format(company_name=company_name)
        data = research_with_grok(prompt, max_tokens=800)
        results["exposures"] = data.get("exposures", "- Not available") if "error" not in data else "- Research failed"
    
    if selected_sections["similar_events"]:
        update_status("Finding similar peer events")
        prompt = SIMILAR_EVENTS_PROMPT_TEMPLATE.format(company_name=company_name, assess_teo="including Tech E&O/privacy/media liability" if assess_teo else "")
        data = research_with_grok(prompt, max_tokens=700, max_retries=4)
        results["similar_events"] = data.get("similar_events", "- None identified") if "error" not in data else "- Research failed"
    
    if selected_sections["vulnerability_scoring"]:
        update_status("Scoring vulnerabilities and overview")
        prompt = VULNERABILITY_PROMPT_TEMPLATE.format(
            company_name=company_name,
            scope="cyber and Tech E&O" if assess_teo else "cyber",
            cyber_categories=json.dumps(CYBER_CATEGORIES)
        )
        data = research_with_grok(prompt, max_tokens=1000)
        if "error" in data:
            results["vulnerability_scores"] = [5] * 6
            results["cyber_risk_overview"] = "- Overview unavailable due to API issue"
            results["estimated_sensitive_records"] = "- Not available"
        else:
            results.update(data)
        
        if assess_teo:
            teo_data = research_with_grok(f"""
            For {company_name} Tech E&O:
            - Rate 1–10 vulnerability for: {json.dumps(TEO_CATEGORIES)}
            - Brief Tech E&O risk bullets.
            JSON: {{"teo_scores": list({len(TEO_CATEGORIES)}), "teo_risk": string}}
            """, max_tokens=800)
            if "error" not in teo_data:
                results.update(teo_data)
            else:
                results["teo_scores"] = [5] * 6
                results["teo_risk"] = "- Tech E&O assessment unavailable"
    
    if selected_sections["targeted_gangs"]:
        update_status("Checking for targeted cyber crime gang activity")
        prompt = TARGETED_GANGS_PROMPT_TEMPLATE
        gang_data = research_with_grok(prompt, max_tokens=800)
        results["targeted_gangs"] = gang_data.get("targeted_gangs", "- Not available") if "error" not in gang_data else "- Research failed"
    else:
        results["targeted_gangs"] = "- Skipped (not selected)"
    
    if selected_sections["loss_scenarios"]:
        update_status("Estimating financial loss scenarios")
        targeted_info = results.get("targeted_gangs", "No evidence")
        prompt = LOSS_ESTIMATION_PROMPT_TEMPLATE.format(
            company_name=company_name,
            revenue_usd=revenue_usd,
            employees=employees,
            targeted_info=targeted_info,
            assess_teo="and Tech E&O" if assess_teo else ""
        )
        loss_data = research_with_grok(prompt, max_tokens=1200)
        if "error" not in loss_data:
            results.update(loss_data)
        else:
            default = {"total": "$5M", "breakdown": "Estimate unavailable - using conservative placeholder"}
            results["cyber_loss_estimates"] = {"poor": {"total": "$50M", "breakdown": default["breakdown"]},
                                               "fair": {"total": "$25M", "breakdown": default["breakdown"]},
                                               "good": {"total": "$10M", "breakdown": default["breakdown"]},
                                               "excellent": {"total": "$3M", "breakdown": default["breakdown"]}}
            if assess_teo:
                results["teo_loss_estimates"] = {"poor": {"total": "$40M", "breakdown": default["breakdown"]},
                                                 "fair": {"total": "$20M", "breakdown": default["breakdown"]},
                                                 "good": {"total": "$8M", "breakdown": default["breakdown"]},
                                                 "excellent": {"total": "$2M", "breakdown": default["breakdown"]}}
    
    progress_bar.progress(1.0)
    status_placeholder.success("**All selected stages complete ✅**")
    
    return results

# ... (UI and display logic unchanged from your last version)

st.set_page_config(page_title="Cyber & Tech E&O Underwriting Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
        max-width: 350px !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] .css-1d391kg {
        width: 350px !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("Cyber & Tech E&O Underwriting Assistant (Beta Version)")
st.markdown("**Developed by Mouad Boughamza** • Company-specific cyber and technology risk analysis • Powered by xAI Grok • Verifiable sources")

# ... (rest of the code identical to previous final version)
