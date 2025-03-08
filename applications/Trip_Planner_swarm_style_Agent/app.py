import streamlit as st
import uuid
from travel_agent_swarm import SwarmPlanner, initialize_data

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "planner" not in st.session_state:
    st.session_state.planner = SwarmPlanner().planner

# Page config
st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

# Title
st.title("✈️ Travel Assistant")

# Sidebar with user info
with st.sidebar:
    st.subheader("Session Information")
    st.write(f"User ID: {st.session_state.user_id}")
    st.write(f"Thread ID: {st.session_state.thread_id}")
    
    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you with your travel plans?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id
            }
        }
        
        events = st.session_state.planner.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config,
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event and len(event["messages"]) > 0:
                last_message = event["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    full_response = last_message.content
                    message_placeholder.write(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit and LangGraph 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
) 