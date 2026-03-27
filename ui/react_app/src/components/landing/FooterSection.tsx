export default function Footer() {
  return (
    <footer className="fixed bottom-0 left-0 w-full z-50 
                       bg-black/30 backdrop-blur-xl 
                       border-t border-white/10">

      <div className="flex items-center justify-between 
                      px-6 md:px-10 py-3 
                      text-sm text-gray-400">

        {/* Left */}
        <p className="opacity-80 hover:opacity-100 transition">
          © 2026 NCERT Companion
        </p>

        {/* Center */}
        <p className="text-center">
          Made with ❤️ by{" "}
          <span className="text-white font-medium hover:text-orange-400 transition">
            Anand Raj
          </span>
        </p>

        {/* Right (Icons from public folder) */}
        <div className="flex items-center gap-4">

          <a
            href="https://github.com/anan5093"
            target="_blank"
            rel="noopener noreferrer"
            className="group"
          >
            <img
              src="/github_3291695.png"
              className="w-7 h-7 md:w-8 md:h-8 
                         transition-all duration-300 
                         group-hover:scale-125 
                         group-hover:rotate-6 
                         opacity-80 group-hover:opacity-100"
            />
          </a>

          <a
            href="https://www.linkedin.com/in/anand-raj-006a41217/"
            target="_blank"
            rel="noopener noreferrer"
            className="group"
          >
            <img
              src="/linkedin_3670129.png"
              className="w-7 h-7 md:w-8 md:h-8 
                         transition-all duration-300 
                         group-hover:scale-125 
                         group-hover:-rotate-6 
                         opacity-80 group-hover:opacity-100"
            />
          </a>

        </div>

      </div>
    </footer>
  );
}
