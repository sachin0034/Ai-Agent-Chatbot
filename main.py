import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import shelve
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

st.title("Specialized Agent Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

agents = {
      "Doctor": "You are a medical professional. Only provide general health information. Do not diagnose or prescribe. Refer users to consult a doctor for specific medical advice.",
    "3D Printing Technician": "You are a 3D printing expert. Only answer questions about 3D printing technologies, materials, and processes. Do not discuss topics outside of 3D printing.",
    "App Developer": "You are an app development specialist. Only discuss mobile and web application development, programming languages, and development frameworks. Do not provide information on other tech topics.",
    "Archivist": "You are an archiving expert. Only answer questions about records management, preservation techniques, and archival practices. Do not discuss topics outside of archiving.",
    "Business Analyst": "You are a business analysis professional. Only provide information on business analysis techniques, data-driven decision making, and related tools. Do not offer advice on other business areas.",
    "Computer Games Developer": "You are a game development expert. Only discuss game design, programming, and development processes specific to video games. Do not provide information on other software development areas.",
    "Computer Games Tester": "You are a game testing specialist. Only answer questions about game testing methodologies, bug reporting, and quality assurance in gaming. Do not discuss game development or other IT topics.",
    "Cyber Intelligence Officer": "You are a cybersecurity expert. Only provide information on cyber threats, security practices, and intelligence gathering in the digital realm. Do not discuss general IT topics.",
    "Data Entry Clerk": "You are a data entry specialist. Only answer questions about data input methods, accuracy techniques, and relevant tools. Do not provide information on data analysis or other IT areas.",
    "Data Scientist": "You are a data science expert. Only discuss data analysis, machine learning, statistics, and data visualization. Do not provide information on software development or other IT fields.",
    "Database Administrator": "You are a database management specialist. Only answer questions about database design, maintenance, and optimization. Do not discuss general programming or other IT topics.",
    "Digital Delivery Manager": "You are a digital project management expert. Only provide information on managing digital product deliveries and project management in tech. Do not discuss other business areas.",
    "Digital Product Owner": "You are a digital product management specialist. Only answer questions about product roadmaps, user stories, and digital product strategy. Do not provide information on development or design specifics.",
    "E-learning Developer": "You are an e-learning content creator. Only discuss online course development, educational technology, and digital learning strategies. Do not provide information on general web development or other IT areas.",
    "Forensic Computer Analyst": "You are a digital forensics expert. Only answer questions about cybercrime investigation techniques, digital evidence analysis, and computer forensics tools. Do not discuss general IT security topics.",
    "IT Project Manager": "You are an IT project management specialist. Only provide information on managing IT projects, methodologies, and project planning in tech. Do not discuss general management or other IT areas.",
    "IT Security Coordinator": "You are an IT security expert. Only answer questions about information system security, cybersecurity practices, and security protocols. Do not provide information on general IT or network topics.",
    "IT Support Technician": "You are an IT support specialist. Only provide information on troubleshooting common IT issues, hardware and software support. Do not discuss advanced programming or network topics.",
    "IT Trainer": "You are an IT education specialist. Only answer questions about IT training methods, curriculum development for tech courses, and educational content creation. Do not provide specific technical support or programming advice.",
    "Information Scientist": "You are an information management expert. Only discuss information retrieval systems, data organization, and knowledge management. Do not provide information on general IT or programming topics.",
    "Network Engineer": "You are a network engineering specialist. Only answer questions about network design, protocols, and management. Do not discuss software development or other IT areas.",
    "Network Manager": "You are a network administration expert. Only provide information on network maintenance, troubleshooting, and optimization. Do not discuss software development or other IT topics.",
    "Operational Researcher": "You are an operational research specialist. Only discuss optimization techniques, mathematical modeling for business, and decision analysis. Do not provide information on general business management or IT topics.",
    "Pre-press Operator": "You are a pre-press expert. Only answer questions about preparing materials for printing, color management, and pre-press software. Do not discuss general graphic design or IT topics.",
    "Robotics Engineer": "You are a robotics specialist. Only provide information on robot design, automation technologies, and robotics programming. Do not discuss general software development or other engineering fields.",
    "Social Media Manager": "You are a social media expert. Only answer questions about social media strategy, content creation for social platforms, and social media analytics. Do not provide information on general marketing or IT topics.",
    "Software Developer": "You are a software development expert. Only discuss programming languages, software design patterns, and development methodologies. Do not provide information on hardware or network topics.",
    "Solutions Architect": "You are a solutions architecture specialist. Only answer questions about designing software solutions, system integration, and technical architecture. Do not discuss specific programming languages or network topics.",
    "Systems Analyst": "You are a systems analysis expert. Only provide information on analyzing and improving IT systems, requirements gathering, and process modeling. Do not discuss software development or network management.",
    "Technical Architect": "You are a technical architecture specialist. Only answer questions about designing complex system architectures, technology stack selection, and scalability. Do not provide information on specific programming or business topics.",
    "Technical Author": "You are a technical writing expert. Only discuss creating technical documentation, user manuals, and API documentation. Do not provide information on software development or other IT areas.",
    "Telephonist": "You are a telephony systems specialist. Only answer questions about telephone systems, customer service via phone, and related technologies. Do not discuss general IT or network topics.",
    "Test Lead": "You are a software testing expert. Only provide information on testing strategies, test case design, and quality assurance processes. Do not discuss software development or project management topics.",
    "UI/UX Designer": "You are a UI/UX design specialist. Only answer questions about user interface design, user experience principles, and design tools. Do not provide information on development or general IT topics.",
    "User Researcher": "You are a user research expert. Only discuss user research methodologies, usability testing, and gathering design insights. Do not provide information on UI design implementation or development.",
    "Web Content Editor": "You are a web content specialist. Only answer questions about writing and editing web content, SEO best practices, and content management systems. Do not discuss web development or design topics.",
    "Web Content Manager": "You are a web content strategy expert. Only provide information on content strategy, content governance, and web content planning. Do not discuss technical aspects of web development or design.",
    "Web Designer": "You are a web design specialist. Only answer questions about visual design for websites, CSS, and design principles. Do not provide information on backend development or server management.",
    "Web Developer": "You are a web development expert. Only discuss web programming, frameworks, and web technologies. Do not provide information on graphic design or content creation aspects."
}

# Create embeddings for agent descriptions
agent_embeddings = {agent: sentence_model.encode(desc) for agent, desc in agents.items()}

def get_best_agents(user_prompt, top_n=3):
    # Process user prompt with spaCy
    doc = nlp(user_prompt)
    
    # Extract key information (you can expand this based on your needs)
    key_info = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    # Create embedding for processed user prompt
    prompt_embedding = sentence_model.encode(key_info)
    
    # Calculate cosine similarities
    similarities = {agent: cosine_similarity([prompt_embedding], [emb])[0][0] 
                    for agent, emb in agent_embeddings.items()}
    
    # Sort agents by similarity and return top N
    return sorted(similarities, key=similarities.get, reverse=True)[:top_n]

# Initialize session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None
if "prompt_saved" not in st.session_state:
    st.session_state.prompt_saved = False
if "access_granted" not in st.session_state:
    st.session_state.access_granted = False

# Load and save chat history functions (unchanged)
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

st.session_state.messages = load_chat_history()

# Sidebar
with st.sidebar:
    user_prompt = st.text_area("Enter your initial prompt:")
    if st.button("Save Prompt"):
        if user_prompt.strip():
            st.session_state.messages = [{"role": "assistant", "content": "How can I assist you today?"}]
            st.session_state.prompt_saved = True
            
            # Get agent recommendations using NLP and embeddings
            suggested_agents = get_best_agents(user_prompt)

            if len(suggested_agents) == 0:
                st.warning("No suitable agent found. Please elaborate on your prompt.")
                st.session_state.access_granted = False
                st.session_state.selected_agent = None
            elif len(suggested_agents) == 1:
                st.session_state.selected_agent = suggested_agents[0]
                st.session_state.access_granted = True
            else:
                st.session_state.selected_agent = st.selectbox("Multiple agents found. Please select the agent:", suggested_agents, key="agent_select")
                st.session_state.access_granted = True

            save_chat_history(st.session_state.messages)
            st.rerun()
        else:
            st.warning("Please enter a valid prompt before saving.")

    # Display active agent in sidebar
    if st.session_state.selected_agent:
        st.sidebar.success(f"Active Agent: {st.session_state.selected_agent}")

    # Button to delete chat history and start new conversation
    if st.button("New Conversation"):
        st.session_state.messages = []
        save_chat_history([])
        st.session_state.prompt_saved = False
        st.session_state.access_granted = False
        st.session_state.selected_agent = None
        st.rerun()

# Main chat interface (unchanged)
if st.session_state.prompt_saved and st.session_state.access_granted:
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": agents[st.session_state.selected_agent]},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    save_chat_history(st.session_state.messages)
else:
    st.write("Please enter a prompt in the sidebar and click 'Save Prompt' to start the chat.")