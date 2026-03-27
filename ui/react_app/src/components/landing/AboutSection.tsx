export default function AboutSection() {
  return (
    <div id="about" className="px-16 py-20 flex flex-col md:flex-row items-center gap-12">

      {/* LEFT → IMAGE */}
      <div className="flex justify-center">
        <img
          src="/Anand-2.jpeg"   // 👈 put image in public folder
          alt="Anand Raj"
          className="w-64 h-64 object-cover rounded-full shadow-xl border-4 border-orange-500"
        />
      </div>

      {/* RIGHT → TEXT */}
      <div className="max-w-xl">

        <h2 className="text-4xl font-bold text-gradient-animated">
          About the Developer
        </h2>

        <p className="text-gray-600 mt-6 leading-relaxed">
          Hi, I'm <span className="font-semibold text-black">Anand Raj</span>, a B.Tech Computer Science student at Galgotias University (2022–2026). 
          I am passionate about building AI-powered systems that solve real-world problems in education. 
          This project, an NCERT RAG-based AI tutor, is inspired by research in intelligent tutoring systems and aims to provide 
          accurate, context-aware learning support using retrieval-augmented generation. 
          My focus is on combining machine learning, system design, and user-centric interfaces to create scalable and impactful AI products.
        </p>

        {/* TECH TAGS */}
        <div className="flex flex-wrap gap-3 mt-6">
          <span className="px-3 py-1 border rounded-full text-sm">Python</span>
          <span className="px-3 py-1 border rounded-full text-sm">FastAPI</span>
          <span className="px-3 py-1 border rounded-full text-sm">FAISS</span>
          <span className="px-3 py-1 border rounded-full text-sm">RAG Architecture</span>
          <span className="px-3 py-1 border rounded-full text-sm">React</span>
        </div>

      </div>

    </div>
  );
}
