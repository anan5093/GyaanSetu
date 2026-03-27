// src/components/landing/FreedomBackground.tsx

import { motion } from "framer-motion";

export default function FreedomBackground() {
  const images = [
    "/bhagat.jpg",
    "/revolution.jpg",
    "/savarkar.jpg",
    "/bharatmata.jpg",
    "/quote.jpg",
  ];

  return (
    <div className="relative w-full flex flex-col items-center mt-16">

      {/* 🖼 Horizontal Image Row */}
      <div className="flex gap-6 justify-center items-center flex-wrap mt-10">

        {images.map((src, i) => (
          <motion.img
            key={i}
            src={src}
            className="w-40 h-52 object-cover rounded-xl shadow-lg opacity-80"

            // Start hidden
            initial={{ opacity: 0, y: 40 }}

            // Appear one by one
            animate={{ opacity: 0.9, y: 0 }}

            transition={{
              delay: i * 0.6,   // ⬅️ slower sequence
              duration: 1.2,
              ease: "easeOut",
            }}
          />
        ))}

      </div>
    </div>
  );
}
