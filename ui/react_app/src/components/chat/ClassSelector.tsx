type Props = {
  onSelect: (cls: number) => void;
};

export default function ClassSelector({ onSelect }: Props) {
  return (
    <div className="animate-fadeInUp bg-white/10 backdrop-blur-lg p-6 rounded-2xl shadow-xl text-center max-w-md">

      {/* AI Message */}
      <div className="bg-gray-800 px-4 py-2 rounded-xl mb-4 inline-block">
        🤖 Hello! Which class are you studying in?
      </div>

      {/* Buttons */}
      <div className="flex gap-3 justify-center flex-wrap">
        {[9, 10, 11, 12].map((cls) => (
          <button
            key={cls}
            onClick={() => onSelect(cls)}
            className="px-4 py-2 bg-orange-500 hover:bg-orange-600 rounded-full transition"
          >
            Class {cls}
          </button>
        ))}
      </div>
    </div>
  );
}
