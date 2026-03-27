import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

const messages = [
  { role: "user", text: "What is force?" },
  { role: "ai", text: "Force is a push or pull acting on an object." },
  { role: "user", text: "Give example" },
  { role: "ai", text: "Pushing a door or pulling a rope are examples of force." }
];

export default function ChatPreview() {
  const [visibleMessages, setVisibleMessages] = useState<any[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let i = 0;
    let timeout: any;

    const runChat = () => {
      // 🔁 Reset loop cleanly
      if (i >= messages.length) {
        i = 0;
        setVisibleMessages([]);
      }

      const msg = messages[i];

      // 👤 USER MESSAGE
      if (msg.role === "user") {
        setVisibleMessages((prev) => [...prev, msg]);
        i++;
        timeout = setTimeout(runChat, 1200);
      }

      // 🤖 AI MESSAGE
      else {
        setIsTyping(true);

        timeout = setTimeout(() => {
          setIsTyping(false);
          setVisibleMessages((prev) => [...prev, msg]);
          i++;
          timeout = setTimeout(runChat, 1500);
        }, 800);
      }
    };

    runChat();

    return () => clearTimeout(timeout);
  }, []);

  // ✅ Scroll ONLY inside chat box
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [visibleMessages, isTyping]);

  return (
    <div
      ref={containerRef}
      className="w-80 h-96 bg-white rounded-xl shadow-xl p-4 flex flex-col gap-3 overflow-y-auto"
    >
      {visibleMessages.map((msg, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className={`max-w-[80%] px-3 py-2 rounded-lg text-sm ${
            msg.role === "user"
              ? "bg-orange-500 text-white self-end"
              : "bg-gray-100 text-gray-800 self-start"
          }`}
        >
          {msg.text}
        </motion.div>
      ))}

      {/* 🤖 Typing indicator */}
      {isTyping && (
        <motion.div
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ repeat: Infinity, duration: 1 }}
          className="text-xs text-gray-400"
        >
          AI is typing...
        </motion.div>
      )}
    </div>
  );
}
