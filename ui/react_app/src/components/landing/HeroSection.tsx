import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom"; // Added missing hook
import ChatPreview from "./ChatPreview";

export default function HeroSection() {
  const navigate = useNavigate(); // Initialized navigate

  return (
    <section id="hero" className="min-h-screen">
    <div className="flex items-center justify-between px-16 py-20">

      {/* ================= LEFT ================= */}
      <motion.div
        initial={{ x: -60, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="max-w-xl"
      >
        {/* Badge */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-orange-100 text-orange-600 px-4 py-1 rounded-full inline-block mb-4"
        >
          ✨ AI-Powered Education Assistant
        </motion.p>

        {/* Heading - Fixed the broken tag here */}
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-5xl font-bold text-gray-900 leading-tight"
        >
          NCERT Learning Companion
        </motion.h1>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-gray-600 mt-6"
        >
          Ask questions, get instant answers with context from NCERT books.
        </motion.p>

        {/* Buttons */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={{
            hidden: {},
            visible: {
              transition: { staggerChildren: 0.2 }
            }
          }}
          className="flex gap-4 mt-8"
        >
          <motion.button
            onClick={() => navigate("/chat")}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="bg-orange-500 text-white px-6 py-3 rounded-full shadow-md"
          >
            Start Learning →
          </motion.button>

          <motion.button
            variants={{
              hidden: { opacity: 0, y: 20 },
              visible: { opacity: 1, y: 0 }
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="border border-orange-500 text-orange-500 px-6 py-3 rounded-full"
          >
            Learn More
          </motion.button>
        </motion.div>
      </motion.div>

      {/* ================= RIGHT (CHAT PREVIEW) ================= */}
      <motion.div
        initial={{ x: 60, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="hidden md:flex justify-center items-center"
      >
        <motion.div
          animate={{ y: [0, -12, 0] }}
          transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
        >
          <ChatPreview />
        </motion.div>
      </motion.div>

    </div>
   
  </section> 
  );
}
