import { useState, useEffect } from "react";
import ChatBackground from "../components/chat/ChatBackground";
import ClassSelector from "../components/chat/ClassSelector";
import ChatWindow from "../components/chat/ChatWindow";
import Navbar from "../components/landing/Navbar";

export default function Chat() {
  const [selectedClass, setSelectedClass] = useState<number | null>(null);
  const [showSelector, setShowSelector] = useState(false);
  const [fadeIn, setFadeIn] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSelector(true);
      setFadeIn(true);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="relative min-h-screen text-white overflow-hidden flex flex-col">

      {/* Background */}
      <ChatBackground />

      {/* Navbar */}
      <div className="relative z-20">
        <Navbar />
      </div>

      {/* Content */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-6">

        {/* Class Selector */}
        {!selectedClass && showSelector && (
          <div
            className={`transition-all duration-700 ease-out ${
              fadeIn ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            <ClassSelector onSelect={setSelectedClass} />
          </div>
        )}

        {/* Chat Window */}
        {selectedClass && (
          <div className="w-full max-w-5xl h-full flex flex-col animate-fadeIn">
            <ChatWindow selectedClass={selectedClass} />
          </div>
        )}

      </div>
    </div>
  );
}
