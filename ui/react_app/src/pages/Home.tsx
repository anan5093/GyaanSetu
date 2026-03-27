import Navbar from "../components/landing/Navbar";
import HeroSection from "../components/landing/HeroSection";
import FeaturesSection from "../components/landing/FeaturesSection";
import VisionSection from "../components/landing/VisionSection";
import AboutSection from "../components/landing/AboutSection";
import Footer from "../components/landing/FooterSection";

import StorySection from "../components/landing/StorySection";
import FreedomBackground from "../components/landing/FreedomBackground";
import AIGoodSection from "../components/landing/AIGoodSection";

export default function Home() {
  return (
    <div className="relative min-h-screen overflow-hidden">

      {/* 🌐 Navbar */}
      <Navbar />

      {/* 🧠 Hero */}
      <HeroSection />

      {/* 📖 Story */}
      <StorySection />

      {/* 🇮🇳 Freedom Section (NOW CORRECT PLACE) */}
      <FreedomBackground />

      {/* ⚡ Features */}
      <FeaturesSection />

      {/* 🤖 AI for Good */}
      <AIGoodSection />

      {/* 🎯 Vision */}
      <VisionSection />

      {/* 📘 About */}
      <AboutSection />

      {/* 🔻 Footer */}
      <Footer />

    </div>
  );
}
