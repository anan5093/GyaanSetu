// src/components/landing/AIGoodSection.tsx

export default function AIGoodSection() {
  return (
    <section className="py-24 px-6 text-center">

      {/* Heading */}
      <h2 className="text-3xl md:text-4xl font-semibold">
        🌍 AI for Good — Education for All
      </h2>

      {/* Description */}
      <p className="text-gray-400 mt-4 max-w-2xl mx-auto">
        Personalized learning powered by AI, grounded in NCERT curriculum,
        accessible to every student — anytime, anywhere.
      </p>

      {/* 🔥 Feature Grid */}
      <div className="mt-12 grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">

        <div className="bg-white/5 p-6 rounded-xl backdrop-blur-md border border-white/10">
          <h3 className="text-lg font-semibold mb-2">📚 Curriculum Aligned</h3>
          <p className="text-gray-400 text-sm">
            Answers strictly based on NCERT textbooks to ensure accuracy and trust.
          </p>
        </div>

        <div className="bg-white/5 p-6 rounded-xl backdrop-blur-md border border-white/10">
          <h3 className="text-lg font-semibold mb-2">⚡ Instant Doubt Solving</h3>
          <p className="text-gray-400 text-sm">
            Get explanations instantly with AI-powered retrieval and reasoning.
          </p>
        </div>

        <div className="bg-white/5 p-6 rounded-xl backdrop-blur-md border border-white/10">
          <h3 className="text-lg font-semibold mb-2">🌐 Accessible Anywhere</h3>
          <p className="text-gray-400 text-sm">
            Learn anytime, anywhere — no dependency on tutors or schedules.
          </p>
        </div>

      </div>

    </section>
  );
}
