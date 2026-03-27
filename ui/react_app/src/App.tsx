import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Home from "./pages/Home";
import Chat from "./pages/Chat";
import Evaluation from "./pages/Evaluation"; // ⭐ NEW

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/evaluation" element={<Evaluation />} /> {/* ⭐ NEW */}
      </Routes>
    </Router>
  );
}
