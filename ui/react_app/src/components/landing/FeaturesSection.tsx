import { motion } from "framer-motion";

const features = [
  "NCERT Content",
  "FAISS Vector Store",
  "OpenRouter LLM",
  "Smart Chunking",
  "Real-time Retrieval",
  "Class-wise Queries"
];

export default function FeaturesSection() {
  return (
    <section id = "features" className="min-h-screen">
    <div className="px-16 py-20">

      <h2 className="text-4xl font-bold text-center mb-12 text-gradient-animated">
        Everything You Need
      </h2>

      <div className="grid md:grid-cols-3 gap-8">
        {features.map((f, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 60 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.15 }}
            whileHover={{ scale: 1.05 }}
            className="p-6 rounded-xl bg-white shadow-md"
          >
            <div className="text-orange-500 text-3xl mb-4">📘</div>
            <h3 className="text-xl font-semibold">{f}</h3>
            <p className="text-gray-600 mt-2">
              AI-powered learning with RAG pipeline
            </p>
          </motion.div>
        ))}
      </div>

    </div>
   </section> 
  );
}
