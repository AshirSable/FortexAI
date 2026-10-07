import { useState, useRef, useEffect } from "react";
import "./App.css";

const getCurrentTime = () => {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
};

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hey! I'm your Personal Food Assistant. What are you craving today?",
      time: getCurrentTime(),
    },
  ]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (customMessage) => {
    const textToSend = (customMessage || input).trim();
    if (!textToSend || loading) return;

    const userMessage = {
      sender: "user",
      text: textToSend,
      time: getCurrentTime(),
    };

    // 1. Immediately append user message to local state
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
      // 2. Format history into { role, content } pairs for the backend
      const formattedHistory = updatedMessages.map((m) => ({
        role: m.sender === "user" ? "user" : "assistant",
        content: m.text,
      }));

      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: formattedHistory,
          message: textToSend, // Fallback field
        }),
      });

      if (!res.ok) throw new Error("API request failed");
      const data = await res.json();

      // 3. Append assistant response
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.response,
          time: getCurrentTime(),
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "I couldn't reach the food servers. Please try again!",
          time: getCurrentTime(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        sender: "bot",
        text: "How else can I assist your order today?",
        time: getCurrentTime(),
      },
    ]);
  };

  return (
    <div className="web-chat-layout">
      <div className="chat-card">
        {/* Header */}
        <header className="chat-header">
          <div className="brand-info">
            <div className="brand-badge">✨</div>
            <div className="brand-text">
              <h2>CraveAI</h2>
              <span className="brand-status">Online Concierge</span>
            </div>
          </div>
          <div className="header-actions">
            <button className="clear-chat-btn" onClick={handleClearChat}>
              Clear Chat
            </button>
          </div>
        </header>

        {/* Chat Feed */}
        <main className="chat-feed">
          <div className="chat-feed-inner">
            {messages.map((msg, index) => (
              <div key={index} className={`message-row ${msg.sender}`}>
                {msg.sender === "bot" && (
                  <div className="avatar bot-avatar">✨</div>
                )}

                <div className="message-container">
                  <div className={`message-bubble ${msg.sender}`}>
                    <p>{msg.text}</p>
                  </div>
                  <span className="message-timestamp">{msg.time}</span>
                </div>

                {msg.sender === "user" && (
                  <div className="avatar user-avatar">👤</div>
                )}
              </div>
            ))}

            {loading && (
              <div className="message-row bot">
                <div className="avatar bot-avatar">✨</div>
                <div className="message-container">
                  <div className="message-bubble bot typing-box">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </main>

        {/* Input Dock */}
        <footer className="chat-footer">
          <div className="input-bar">
            <span className="input-adornment">✨</span>
            <input
              type="text"
              placeholder="Ask CraveAI about recommendations, menus, or delivery..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !loading) handleSend();
              }}
            />
            <button
              className="send-button"
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              aria-label="Send Message"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;