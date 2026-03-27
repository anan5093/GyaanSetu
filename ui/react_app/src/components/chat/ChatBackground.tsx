export default function ChatBackground() {
  return (
    <>
      {/* Base Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#0f172a] via-[#1e293b] to-[#7c2d12]" />

      {/* Orange Glow (Top Right) */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-orange-500/20 blur-3xl rounded-full" />

      {/* Blue Glow (Bottom Left) */}
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-500/20 blur-3xl rounded-full" />
    </>
  );
}
