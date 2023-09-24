import openai
import streamlit as st
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

st.set_page_config(layout="wide")

@st.cache_data
def measure_context_sensitivity(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert legal advisor. Give 2-4 points to explain context that might be missing in the given query to give the best legal advice and give a score out of 10. Include an understanding of legal standard and industry standard."},
            {"role": "user", "content": query}
        ]
    )
    markdown_content = response['choices'][0]['message']['content']
    return markdown_content

@st.cache_data
def measure_realtime_adaptability(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert legal advisor with model cut-off date of 20th April 2023. Give 2-4 points to explain given the model cutoff date why the model is not able to give the best legal advice and give a score out of 10."},
            {"role": "user", "content": query}
        ]
    )
    markdown_content = response['choices'][0]['message']['content']
    return markdown_content

@st.cache_data
def get_legal_advice(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert legal advisor. Give 2-4 points based on any given query."},
            {"role": "user", "content": query}
        ]
    )
    markdown_content = response['choices'][0]['message']['content']
    return markdown_content

@st.cache_data
def get_indemnity_table(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert legal advisor."},
            {"role": "user", "content": query + """
For the given legal advice, please provide a detailed analysis. 
1. Identify each potential legal liability.
2. For each liability, list any potential ambiguities.
3. Provide a table in Github-flavored markdown summarizing risk level, liability, its ambiguities, and detailed associated penalties including amount of dollar damage and potential jail time if applicable for the lawyer giving the advice if they get it wrong.
Only return the table.
Table:"""}
        ]
    )
    markdown_content = response['choices'][0]['message']['content']
    return markdown_content


st.header("Pondus AI - Evaluation Framework For Lawyers In Legal Firms")
query = st.text_area("Enter your query here", 
    value="""What steps should your company take to ensure compliance with data privacy laws and protect against potential fines and legal liabilities in your industry?""")


if st.button('Submit'):
    col1, col2 = st.columns(2)
    with st.spinner("LegalGPT Generating Answer..."):
        answer = get_legal_advice(query)
    with col1:
        st.markdown("LegalGPT Advice:")
        st.markdown(answer)
    with col2:
        st.markdown("Legal Liability Evaluation:")
        with st.spinner('Analyzing response...'):
            response = get_indemnity_table(answer)
        st.markdown(response)

    test_case = LLMTestCase(
        query=query,
        output=answer[:500],
        expected_output="-",
        context="-"
    )
    st.header("Context Sensitivity")
    context_sensitivity = measure_context_sensitivity(query)
    st.markdown("For the given context, please note that it only follows legal standards and requires further industry standards to be added to give the best legal advice.")
    st.markdown(context_sensitivity)

    st.header("Real-time Adapability")
    realtime_adaptability = measure_realtime_adaptability(query)

    st.markdown(realtime_adaptability)

    metric = AnswerRelevancyMetric()
    answer_relevancy_score = metric.measure(
        test_case=test_case
    )
    st.markdown("Answer Relevancy: "+ str(answer_relevancy_score))
