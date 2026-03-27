import { useNavigate, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

export default function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const [active, setActive] = useState("hero");

  const sections = ["hero", "features", "about", "vision"];

  // 🔥 Scroll detection (ACTIVE NAV)
  useEffect(() => {
    if (location.pathname !== "/") return;

    const observers: IntersectionObserver[] = [];

    sections.forEach((id) => {
      const el = document.getElementById(id);

      if (!el) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setActive(id);
          }
        },
        { threshold: 0.6 }
      );

      observer.observe(el);
      observers.push(observer);
    });

    return () => observers.forEach((obs) => obs.disconnect());
  }, [location.pathname]);

  // 🔁 Navigation + Scroll
  const goToSection = (id: string) => {
    if (location.pathname === "/") {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    } else {
      navigate("/");

      setTimeout(() => {
        document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
      }, 400);
    }
  };

  return (
    <div className="fixed top-0 w-full z-50 backdrop-blur-xl bg-black/30 border-b border-white/10">
      <div className="flex justify-between items-center px-10 py-4 text-white">

        {/* Logo */}
        <div
          className="font-semibold text-lg cursor-pointer"
          onClick={() => navigate("/")}
        >
          ✨ <span className="text-orange-500">NCERT</span> Companion
        </div>

        {/* Nav */}
        <div className="flex gap-8 items-center relative">

          {/* Landing Sections */}
          {sections.map((sec) => (
            <button
              key={sec}
              onClick={() => goToSection(sec)}
              className={`relative pb-1 capitalize transition ${
                active === sec && location.pathname === "/"
                  ? "text-orange-400"
                  : "hover:text-orange-300"
              }`}
            >
              {sec}

              {/* underline */}
              <span
                className={`absolute left-0 bottom-0 h-[2px] bg-orange-400 transition-all duration-300 ${
                  active === sec && location.pathname === "/"
                    ? "w-full"
                    : "w-0"
                }`}
              />
            </button>
          ))}

          {/* 🔥 NEW: Evaluation Page */}
          <button
            onClick={() => navigate("/evaluation")}
            className={`relative pb-1 transition ${
              location.pathname === "/evaluation"
                ? "text-orange-400"
                : "hover:text-orange-300"
            }`}
          >
            Evaluation

            <span
              className={`absolute left-0 bottom-0 h-[2px] bg-orange-400 transition-all duration-300 ${
                location.pathname === "/evaluation" ? "w-full" : "w-0"
              }`}
            />
          </button>

          {/* Chat CTA */}
          <button
            onClick={() => navigate("/chat")}
            className={`ml-4 px-5 py-2 rounded-full transition-all duration-300 hover:scale-105 shadow-lg ${
              location.pathname === "/chat"
                ? "bg-orange-600"
                : "bg-orange-500 hover:bg-orange-600"
            }`}
          >
            Start Chat
          </button>

        </div>
      </div>
    </div>
  );
}
