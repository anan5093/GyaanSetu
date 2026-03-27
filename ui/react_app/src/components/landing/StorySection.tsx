// src/components/landing/StorySection.tsx

import { motion } from "framer-motion";

export default function StorySection() {
  return (
    <section className="py-32 text-center max-w-4xl mx-auto">

      <motion.h2
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="text-4xl font-bold"
      >
        From Freedom Fighters to Future Learners
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-6 text-gray-400"
      >
        The same spirit that once fought for independence
        now drives a new revolution — education powered by AI.
      </motion.p>

    </section>
  );
}
