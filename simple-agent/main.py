import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load env vars
load_dotenv()

# === LLM setup ===
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# === Firecrawl MCP setup ===
server_params = StdioServerParameters(
    command="npx",
    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
    args=["firecrawl-mcp"]
)

# --- Streamlit UI ---
st.set_page_config(page_title="Client Research Agent", layout="wide")
st.title("📑 Research & Dossier Agent")
st.write("Generate client dossiers with web intelligence.")

# Input box
client_name = st.text_input("Enter Client/Customer Name", "")

# Run agent button
if st.button("Generate Dossier") and client_name:
    with st.spinner("🔍 Researching and generating dossier..."):

        async def run_agent():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    agent = create_react_agent(model, tools)

                    messages = [
                        SystemMessage(
                            content=(
                                "You are a **Research & Dossier Agent** for **Tata Consultancy Services (TCS)**.\n"
                                "Your job is to prepare a structured dossier for upcoming client/customer visits.\n"
                                "Use Firecrawl tools to search the web, crawl websites, and extract recent updates.\n\n"
                                "⚠️ Important: Always format your output in **Markdown** with clear headings and emojis.\n"
                                "Use the following structure:\n\n"
                                "# 📑 Client Dossier: [Client Name]\n\n"
                                "## 🏢 Background\n"
                                "- Overview of the company/client\n\n"
                                "## 🤝 Past/Current Relations with TCS\n"
                                "- Known partnerships, deals, or interactions with TCS\n\n"
                                "## 📰 Recent News & Updates\n"
                                "- Latest external news, press releases, industry mentions\n\n"
                                "## 🚀 Opportunities\n"
                                "- Potential areas for collaboration or expansion\n\n"
                                "## ⚠️ Risks/Concerns\n"
                                "- Challenges, competition, or red flags\n\n"
                                "## ✅ Recommendations for TCS Team\n"
                                "- Actionable steps for the upcoming visit\n\n"
                                "Keep it professional, structured, and concise."
                            )
                        ),
                        HumanMessage(content=f"Create a dossier for {client_name}.")
                    ]

                    # Run agent
                    response = await agent.ainvoke({"messages": messages})

                    # Extract last AI response
                    last_message = response["messages"][-1]
                    return getattr(last_message, "content", str(last_message))

        # Run async loop inside Streamlit
        dossier = asyncio.run(run_agent())

        # Show result (Markdown rendering)
        st.subheader("📑 Generated Dossier")
        st.markdown(dossier)

