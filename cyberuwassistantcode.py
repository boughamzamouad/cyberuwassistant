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

def generate_radar_chart(scores: List[float], categories: List[str], title: str):
    num = len(categories)
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist() + [0]
    values = scores + [scores[0]]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_title(f"{title}\n(Higher = Greater Vulnerability)", size=14, pad=15)
    return fig

def generate_loss_bar_chart(cyber_estimates: Dict, teo_estimates: Dict = None, company_name: str = ""):
    tiers = ["Poor", "Fair", "Good", "Excellent"]
    cyber_values = []
    for tier in ["poor", "fair", "good", "excellent"]:
        total_str = cyber_estimates.get(tier, {}).get("total", "N/A")
        if total_str in ["N/A", "Unavailable"]:
            cyber_values.append(0)
        else:
            try:
                num_str = total_str.replace("$", "").replace(",", "").strip()
                if 'M' in num_str.upper():
                    num = float(num_str.upper().replace("M", ""))
                else:
                    num = float(num_str) / 1_000_000
                cyber_values.append(max(num, 0))
            except:
                cyber_values.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tiers))
    width = 0.35
    
    bars1 = ax.bar(x, cyber_values, width, label='Cyber', color='steelblue')
    
    if teo_estimates:
        teo_values = []
        for tier in ["poor", "fair", "good", "excellent"]:
            total_str = teo_estimates.get(tier, {}).get("total", "N/A")
            if total_str in ["N/A", "Unavailable"]:
                teo_values.append(0)
            else:
                try:
                    num_str = total_str.replace("$", "").replace(",", "").strip()
                    if 'M' in num_str.upper():
                        num = float(num_str.upper().replace("M", ""))
                    else:
                        num = float(num_str) / 1_000_000
                    teo_values.append(max(num, 0))
                except:
                    teo_values.append(0)
        bars2 = ax.bar(x + width, teo_values, width, label='Tech E&O', color='orange')
    
    ax.set_ylabel('Estimated Loss ($M USD)')
    ax.set_title(f'Estimated Financial Loss Scenarios\n{company_name}')
    ax.set_xticks(x + (width / 2 if teo_estimates else 0))
    ax.set_xticklabels(tiers)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, None)
    
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}M',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if teo_estimates:
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}M',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    return fig

if PDF_AVAILABLE:
    def create_pdf_report(results: Dict, assess_teo: bool, selected_sections: Dict[str, bool]) -> BytesIO:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=1*inch, bottomMargin=1*inch)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='TitleBold', parent=styles['Title'], fontSize=20, alignment=1, spaceAfter=30))
        styles.add(ParagraphStyle(name='Heading1Bold', parent=styles['Heading1'], fontSize=16, spaceBefore=20, spaceAfter=12))
        styles.add(ParagraphStyle(name='Heading2Bold', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10))
        styles.add(ParagraphStyle(name='BodyText', parent=styles['Normal'], fontSize=11, leading=14, spaceAfter=8, leftIndent=20))
        
        story = []
        
        story.append(Paragraph(f"{results['company']} Cyber & Tech E&O Risk Profile", styles['TitleBold']))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Generated by Cyber & Tech E&O Underwriting Assistant (Beta Version)", styles['Normal']))
        story.append(Paragraph("Developed by Mouad Boughamza", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        story.append(Paragraph("Key Metrics", styles['Heading1Bold']))
        story.append(Paragraph(f"Annual Revenue: ${int(float(results['revenue_usd'])):,} USD" if results['revenue_usd'] != "Unknown" else "Annual Revenue: Unknown", styles['BodyText']))
        story.append(Paragraph(f"Employees: {results['employees']}", styles['BodyText']))
        story.append(Spacer(1, 0.5*inch))
        
        if selected_sections["description"]:
            story.append(Paragraph("Operations Description", styles['Heading2Bold']))
            desc = results.get("description", "Not available").replace("- ", "• ")
            story.append(Paragraph(desc, styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["subsidiaries"]:
            story.append(Paragraph("Offices / Subsidiaries", styles['Heading2Bold']))
            subs = results.get("subsidiaries", "None identified").replace("- ", "• ")
            story.append(Paragraph(subs, styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["incidents"]:
            story.append(Paragraph("Reported Incidents (Cyber, Tech E&O, Media & Privacy Liability)", styles['Heading2Bold']))
            inc = results.get("incidents", "No publicly reported incidents found").replace("- ", "• ")
            story.append(Paragraph(inc, styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["exposures"]:
            story.append(Paragraph("Key Cyber Exposures (Ranked by Severity)", styles['Heading2Bold']))
            exp = results.get("exposures", "Not available").replace("- ", "• ")
            story.append(Paragraph(exp, styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["similar_events"]:
            story.append(Paragraph("Similar Peer Events", styles['Heading2Bold']))
            sim = results.get("similar_events", "None identified").replace("- ", "• ")
            story.append(Paragraph(sim, styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["vulnerability_scoring"]:
            story.append(Paragraph("Cyber Risk Overview", styles['Heading2Bold']))
            overview = results.get("cyber_risk_overview", "Not available").replace("- ", "• ")
            story.append(Paragraph(overview, styles['BodyText']))
            story.append(Spacer(1, 0.5*inch))
            
            radar_cyber = generate_radar_chart(results.get("vulnerability_scores", [5]*6), CYBER_CATEGORIES, results['company'])
            cyber_io = BytesIO()
            radar_cyber.savefig(cyber_io, format='png', bbox_inches='tight', dpi=300)
            plt.close(radar_cyber)
            cyber_io.seek(0)
            story.append(Paragraph("Cyber Vulnerability Profile", styles['Heading2Bold']))
            story.append(RLImage(cyber_io, width=6*inch, height=6*inch))
            story.append(Spacer(1, 0.4*inch))
            
            if assess_teo:
                story.append(Paragraph("Tech E&O Risk Overview", styles['Heading2Bold']))
                teo_overview = results.get("teo_risk", "Not available").replace("- ", "• ")
                story.append(Paragraph(teo_overview, styles['BodyText']))
                story.append(Spacer(1, 0.5*inch))
                
                radar_teo = generate_radar_chart(results.get("teo_scores", [5]*6), TEO_CATEGORIES, results['company'])
                teo_io = BytesIO()
                radar_teo.savefig(teo_io, format='png', bbox_inches='tight', dpi=300)
                plt.close(radar_teo)
                teo_io.seek(0)
                story.append(Paragraph("Tech E&O Vulnerability Profile", styles['Heading2Bold']))
                story.append(RLImage(teo_io, width=6*inch, height=6*inch))
                story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["targeted_gangs"]:
            story.append(Paragraph("Targeted Cyber Crime Gang Activity", styles['Heading2Bold']))
            gangs = results.get("targeted_gangs", "Not available").replace("- ", "• ")
            story.append(Paragraph(gangs, styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
        
        if selected_sections["loss_scenarios"]:
            story.append(Paragraph("Estimated Financial Loss Scenarios (Major Incident in USA)", styles['Heading1Bold']))
            
            story.append(Paragraph("Cyber Policy Loss Estimates", styles['Heading2Bold']))
            cyber_loss = results.get("cyber_loss_estimates", {})
            for tier in ["poor", "fair", "good", "excellent"]:
                est = cyber_loss.get(tier, {"total": "N/A", "breakdown": "N/A"})
                story.append(Paragraph(f"{tier.capitalize()} controls: {est['total']} — {est['breakdown']}", styles['BodyText']))
            story.append(Spacer(1, 0.4*inch))
            
            if assess_teo:
                story.append(Paragraph("Tech E&O Policy Loss Estimates", styles['Heading2Bold']))
                teo_loss = results.get("teo_loss_estimates", {})
                for tier in ["poor", "fair", "good", "excellent"]:
                    est = teo_loss.get(tier, {"total": "N/A", "breakdown": "N/A"})
                    story.append(Paragraph(f"{tier.capitalize()} controls: {est['total']} — {est['breakdown']}", styles['BodyText']))
                story.append(Spacer(1, 0.4*inch))
            
            bar_fig = generate_loss_bar_chart(
                cyber_estimates=results.get("cyber_loss_estimates", {}),
                teo_estimates=results.get("teo_loss_estimates") if assess_teo else None,
                company_name=results['company']
            )
            bar_io = BytesIO()
            bar_fig.savefig(bar_io, format='png', bbox_inches='tight', dpi=300)
            plt.close(bar_fig)
            bar_io.seek(0)
            story.append(Paragraph("Loss Scenario Comparison Chart", styles['Heading2Bold']))
            story.append(RLImage(bar_io, width=7.5*inch, height=4.5*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
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
            assess_teo=" and Tech E&O" if assess_teo else ""
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

st.set_page_config(
    page_title="Cyber & Tech E&O Underwriting Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar visibility
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 380px !important;
        min-width: 380px !important;
        max-width: 380px !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Cyber & Tech E&O Underwriting Assistant (Beta Version)")
st.markdown("**Developed by Mouad Boughamza** • Company-specific cyber and technology risk analysis • Powered by xAI Grok • Verifiable sources")

st.info("Enter company details and select the analysis sections you wish to run, then click 'Start Research'.")

with st.sidebar:
    st.header("Company Inputs")
    company_name = st.text_input("Company Name *", key="company_name")
    address = st.text_input("Headquarters Address", placeholder="or leave blank", key="address")
    revenue = st.text_input("Annual Revenue (USD)", placeholder="e.g., 56400000 or 'unknown'", key="revenue")
    employees = st.text_input("Number of Employees", placeholder="or 'unknown'", key="employees")
    assess_teo = st.checkbox("Assess Technology Errors & Omissions?", key="assess_teo")
    
    st.markdown("---")
    st.header("Select Analysis Sections")
    select_all = st.checkbox("Select All Sections", value=True, key="select_all")
    
    selected_sections = {
        "description": st.checkbox("Operations Description", value=select_all, key="sec_description"),
        "subsidiaries": st.checkbox("Offices / Subsidiaries", value=select_all, key="sec_subsidiaries"),
        "incidents": st.checkbox("Reported Incidents (Cyber, Tech E&O, Media & Privacy Liability)", value=select_all, key="sec_incidents"),
        "exposures": st.checkbox("Key Cyber Exposures", value=select_all, key="sec_exposures"),
        "similar_events": st.checkbox("Similar Peer Events", value=select_all, key="sec_similar"),
        "vulnerability_scoring": st.checkbox("Vulnerability Scoring & Cyber Risk Overview", value=select_all, key="sec_vulnerability"),
        "targeted_gangs": st.checkbox("Targeted Cyber Crime Gang Activity", value=select_all, key="sec_gangs"),
        "loss_scenarios": st.checkbox("Financial Loss Scenarios & Charts", value=select_all, key="sec_loss"),
    }
    
    avg_time_per_section = 45
    selected_count = sum(selected_sections.values())
    min_time = selected_count * avg_time_per_section / 60
    max_time = min_time * 1.5
    if selected_count > 0:
        st.info(f"**Estimated wait time:** {min_time:.1f}–{max_time:.1f} minutes for selected sections")
    else:
        st.warning("Please select at least one section")
    
    start = st.button("Start Research", type="primary", disabled=not company_name.strip() or selected_count == 0)

if start:
    try:
        with st.spinner("Initializing research..."):
            results = run_full_analysis(company_name, address, revenue, employees, assess_teo, selected_sections)
        
        st.success("Analysis Complete")
        st.header(f"{results['company']} Risk Profile")
        
        col1, col2 = st.columns(2)
        revenue_display = f"${int(float(results['revenue_usd'])):,}" if results['revenue_usd'] != "Unknown" else "Unknown"
        col1.metric("Annual Revenue", revenue_display)
        col2.metric("Employees", results['employees'])
        
        st.markdown("---")
        
        if selected_sections["description"]:
            st.subheader("Operations Description")
            st.markdown(results.get("description", "- Not available"))
            st.markdown("---")
        
        if selected_sections["subsidiaries"]:
            st.subheader("Offices / Subsidiaries")
            st.markdown(results.get("subsidiaries", "- None identified"))
            st.markdown("---")
        
        if selected_sections["incidents"]:
            st.subheader("Reported Incidents (Cyber, Tech E&O, Media & Privacy Liability)")
            st.markdown(results.get("incidents", "- No publicly reported incidents found"))
            st.markdown("---")
        
        if selected_sections["exposures"]:
            st.subheader("Key Cyber Exposures (Ranked by Severity)")
            st.markdown(results.get("exposures", "- Not available"))
            st.markdown("---")
        
        if selected_sections["similar_events"]:
            st.subheader("Similar Peer Events")
            st.markdown(results.get("similar_events", "- None identified"))
            st.markdown("---")
        
        if selected_sections["vulnerability_scoring"]:
            st.subheader("Cyber Risk Overview")
            st.markdown(results.get("cyber_risk_overview", "- Not available"))
            st.markdown("**Estimated Sensitive Records (PII/PHI/NPI)**")
            st.markdown(results.get("estimated_sensitive_records", "- Not available"))
            st.markdown("---")
            
            st.subheader("Cyber Vulnerability Profile")
            st.pyplot(generate_radar_chart(results.get("vulnerability_scores", [5]*6), CYBER_CATEGORIES, results['company']))
            st.markdown("---")
            
            if assess_teo:
                st.subheader("Tech E&O Risk Overview")
                st.markdown(results.get("teo_risk", "- Not available"))
                st.markdown("---")
                st.subheader("Tech E&O Vulnerability Profile")
                st.pyplot(generate_radar_chart(results.get("teo_scores", [5]*6), TEO_CATEGORIES, results['company']))
                st.markdown("---")
        
        if selected_sections["targeted_gangs"]:
            st.subheader("Targeted Cyber Crime Gang Activity")
            st.markdown(results.get("targeted_gangs", "- Not available"))
            st.markdown("---")
        
        if selected_sections["loss_scenarios"]:
            st.subheader("Estimated Financial Loss Scenarios (Major Incident in USA)")
            
            st.markdown("**Cyber Policy Loss Estimates**")
            cyber_loss = results.get("cyber_loss_estimates", {})
            for tier in ["poor", "fair", "good", "excellent"]:
                est = cyber_loss.get(tier, {"total": "N/A", "breakdown": "N/A"})
                st.markdown(f"- **{tier.capitalize()} controls**: {est['total']}  \n  Breakdown: {est['breakdown']}")
            
            if assess_teo:
                st.markdown("**Tech E&O Policy Loss Estimates**")
                teo_loss = results.get("teo_loss_estimates", {})
                for tier in ["poor", "fair", "good", "excellent"]:
                    est = teo_loss.get(tier, {"total": "N/A", "breakdown": "N/A"})
                    st.markdown(f"- **{tier.capitalize()} controls**: {est['total']}  \n  Breakdown: {est['breakdown']}")
            
            st.markdown("---")
            st.subheader("Loss Scenario Comparison")
            bar_fig = generate_loss_bar_chart(
                cyber_estimates=results.get("cyber_loss_estimates", {}),
                teo_estimates=results.get("teo_loss_estimates") if assess_teo else None,
                company_name=results['company']
            )
            st.pyplot(bar_fig)
            st.markdown("---")
        
        st.subheader("Download Report")
        if PDF_AVAILABLE:
            pdf_buffer = create_pdf_report(results, assess_teo, selected_sections)
            st.download_button(
                label="Download Full Report as PDF",
                data=pdf_buffer,
                file_name=f"{results['company'].replace(' ', '_')}_Cyber_Risk_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.info("PDF download not available. Install ReportLab: `pip install reportlab`")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}. Please check the terminal logs for full details.")

st.caption("Beta Version • Developed by Mouad Boughamza • Optimized for performance • Powered by xAI Grok")
