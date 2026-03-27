export default function VisionSection() {
  return (
    <section id="vision" className="min-h-screen">
    <div className="px-16 py-20">

      <div className="bg-gray-900 text-white p-12 rounded-xl">

        {/* Heading */}
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gradient-animated">
          Inspiration & Case Study
        </h2>

        {/* Content */}
        <div className="mt-8 max-w-4xl mx-auto text-gray-300 space-y-6 text-center">

          <p>
            This project is inspired by real-world research on AI-powered tutoring systems,
            particularly Retrieval-Augmented Generation (RAG) based assistants used in education.
          </p>

          <p>
            Studies like the NeuroBot Teaching Assistant demonstrated that students trust AI tutors
            more when responses are grounded in verified course materials, enabling reliable and
            personalized learning experiences.
          </p>

          <p>
            Research also shows that students primarily use such systems for exam preparation and
            quick concept clarification — making them powerful tools for just-in-time learning.
          </p>

          <p>
            Building on these insights, this project applies the same principles to NCERT curriculum,
            aiming to deliver accurate, context-aware answers while maintaining trust and alignment
            with official textbooks.
          </p>

        </div>

      </div>

    </div>
   </section>
  );
}
