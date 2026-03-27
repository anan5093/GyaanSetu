import { useEffect, useState } from "react";
import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

export default function Evaluation() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const fetchEvaluation = async () => {
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/evaluation");
      const json = await res.json();

      console.log("EVAL DATA:", json); // 🔥 DEBUG

      setData(json.evaluation);
    } catch (err) {
      console.error("Error fetching evaluation:", err);
    }

    setLoading(false);
  };

  useEffect(() => {
    fetchEvaluation();
  }, []);

  return (
    <div className="min-h-screen text-white">

      {/* ✅ Navbar */}
      <Navbar />

      <div className="pt-24 px-10">

        <h1 className="text-3xl font-bold mb-6">
          📊 Evaluation Dashboard
        </h1>

        {loading && <p>Running evaluation...</p>}

        {!data && !loading && (
          <p className="text-gray-400">No data received from backend</p>
        )}

        {data && (
          <div className="space-y-10">

            {/* 🔍 Retrieval */}
            <div className="bg-white/5 p-6 rounded-xl">
              <h2 className="text-xl mb-2">🔍 Retrieval</h2>
              <p>Accuracy: {data.retrieval.accuracy}</p>
            </div>

            {/* 🤖 Generation */}
            <div className="bg-white/5 p-6 rounded-xl">
              <h2 className="text-xl mb-2">🤖 Answer Quality</h2>
              <p>Avg Similarity: {data.generation.avg_similarity}</p>
            </div>

            {/* ⏱ Latency */}
            <div className="bg-white/5 p-6 rounded-xl">
              <h2 className="text-xl mb-2">⏱ Latency</h2>
              <p>Avg: {data.latency.avg_latency.toFixed(2)} sec</p>
            </div>

            {/* 🧠 Embedding Chart */}
            <div className="bg-white/5 p-6 rounded-xl">
              <h2 className="text-xl mb-4">🧠 Embedding Space</h2>

              <ScatterChart width={500} height={300}>
                <XAxis dataKey="x" />
                <YAxis dataKey="y" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter data={data.embedding} />
              </ScatterChart>
            </div>

          </div>
        )}
      </div>

      {/* ✅ Footer */}
      <FooterSection />
    </div>
  );
}
