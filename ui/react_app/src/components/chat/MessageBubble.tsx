export default function MessageBubble({ sender, text }) {
  return (
    <div
      className={`max-w-[65%] px-5 py-3 rounded-2xl shadow-lg animate-slideUp ${
        sender === "user"
          ? "bg-orange-500 ml-auto"
          : "bg-white/10 backdrop-blur-xl border border-white/10"
      }`}
    >
      {text}
    </div>
  );
}
