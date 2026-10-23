# Seed a greeting the first time the panel is opened
if st.session_state.chat_open and not st.session_state.chat_messages:
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": "Hi! 👋 How can I help you? Ask me about your soil results, fertilizer options, or alternative crops.",
    })

# Style the toggle button as a small circular icon
st.markdown("""
<style>
.st-key-chat_toggle_btn button {
    border-radius: 50% !important;
    width: 56px !important;
    height: 56px !important;
    font-size: 22px !important;
    background-color: #2e7d32 !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    padding: 0 !important;
}
.st-key-chat_panel {
    background: white;
    border-radius: 14px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
    padding: 14px 16px 8px 16px;
    border: 1px solid #e0e0e0;
}
.st-key-chat_panel_header {
    background-color: #2e7d32;
    color: white;
    padding: 10px 14px;
    border-radius: 10px;
    margin: -14px -16px 10px -16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Floating toggle icon (bottom-right, always visible) ---
toggle_box = st.container(key="chat_toggle_box")
with toggle_box:
    icon_label = "✕" if st.session_state.chat_open else "💬"
    if st.button(icon_label, key="chat_toggle_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
toggle_box.float("bottom: 24px; right: 24px; z-index: 9999;")

# --- Expandable chat panel ---
if st.session_state.chat_open:
    panel = st.container(key="chat_panel")
    with panel:
        st.markdown('<div class="st-key-chat_panel_header">🌱 Ask me about your soil</div>', unsafe_allow_html=True)

        history_box = st.container(height=300)
        with history_box:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_query = st.chat_input("Ask about your soil...", key="floating_chat_input")

    panel.float(
        "bottom: 90px; right: 24px; width: 340px; max-height: 480px; z-index: 9998;"
    )

    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})

        soil_ctx = (analysis or {}).get("soil_context", "No field analysis has been run yet in this session.")
        plan_ctx = (analysis or {}).get("plan_text") or "No AI plan generated yet."

        # Include recent turns so follow-ups like "what about the second one?" resolve correctly
        history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.chat_messages[-9:-1]
        )

        chat_prompt = f"""
        You are an agricultural advisor assistant helping a farmer act on a soil/crop analysis
        that has already been run. Answer the farmer's question directly and practically.

        Field Analysis Context:
        {soil_ctx}

        Previously Generated Advisory Plan:
        {plan_ctx}

        Conversation so far:
        {history}

        Guidance:
        - If asked about fertilizer brands, give general product *categories* and active-ingredient
          guidance (e.g., "a urea-based N fertilizer (46-0-0)" or "a balanced NPK 19-19-19 blend")
          rather than naming specific commercial brands, since availability varies heavily by region
          and endorsing a brand is not something you can verify. Suggest the farmer check with a
          local agricultural supply store or extension office for specific brands available to them.
        - If asked about alternative crops beyond the top 3, reason using the same field parameters
          (N, P, K, pH, soil type, rainfall, temperature, fertility levels) to suggest other
          agronomically reasonable options, and briefly explain why each could or couldn't work here.
        - Keep answers concise and actionable. Use the field context above rather than generic advice.

        Farmer's Question: {user_query}
        """

        try:
            bot_text, _ = generate_with_retry_and_fallback(chat_prompt)
            st.session_state.chat_messages.append({"role": "assistant", "content": bot_text})
        except Exception as e:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"Sorry, I couldn't get a response right now ({e}). Please try again in a moment.",
            })
        st.rerun()
