import React, { useEffect, useState } from 'react';
import { getHealth, getTopics, postAttempt, getRecommendation } from './api';

export default function App() {
  const [status, setStatus] = useState(null);
  const [topics, setTopics] = useState([]);
  const [studentId, setStudentId] = useState('U1');
  const [selTopic, setSelTopic] = useState('Algebra');
  const [difficulty, setDifficulty] = useState(2);
  const [correct, setCorrect] = useState(1);
  const [recs, setRecs] = useState([]);
  const [policy, setPolicy] = useState('baseline');
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');

  useEffect(() => {
    getHealth().then(setStatus).catch(() => setStatus({ ok: false }));
    getTopics().then((t) => { setTopics(t); if (t?.length) setSelTopic(t[0].topic); });
  }, []);

  const submitAttempt = async () => {
    setMsg('');
    try {
      await postAttempt({
        student_id: studentId.trim(),
        topic: selTopic,
        difficulty: Number(difficulty),
        correct: Number(correct),
      });
      setMsg('Saved attempt.');
    } catch (e) {
      setMsg('Error saving attempt.');
    }
  };

  const fetchRecs = async () => {
    setLoading(true); setMsg('');
    try {
      const data = await getRecommendation(studentId.trim(), 3, policy);
      setRecs(data.next_topics || []);
    } catch (e) {
      setMsg('Error fetching recommendation.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Adaptive Quiz Teacher</h1>
        <span className={"text-sm " + (status?.ok ? "text-green-600" : "text-red-600")}>
          API {status?.ok ? "connected" : "offline"}
        </span>
      </header>

      <section className="bg-white rounded-2xl shadow p-5 space-y-4">
        <h2 className="text-lg font-medium">Log an attempt</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <input className="border rounded px-3 py-2" value={studentId} onChange={e => setStudentId(e.target.value)} placeholder="Student ID" />
          <select className="border rounded px-3 py-2" value={selTopic} onChange={e => setSelTopic(e.target.value)}>
            {topics.map(t => <option key={t.topic} value={t.topic}>{t.topic}</option>)}
          </select>
          <select className="border rounded px-3 py-2" value={difficulty} onChange={e => setDifficulty(Number(e.target.value))}>
            <option value={1}>Difficulty 1</option>
            <option value={2}>Difficulty 2</option>
            <option value={3}>Difficulty 3</option>
          </select>
          <select className="border rounded px-3 py-2" value={correct} onChange={e => setCorrect(Number(e.target.value))}>
            <option value={1}>Correct</option>
            <option value={0}>Incorrect</option>
          </select>
        </div>
        <button onClick={submitAttempt} className="px-4 py-2 rounded bg-blue-600 text-white">Save attempt</button>
        {msg && <div className="text-sm text-gray-700">{msg}</div>}
      </section>

      <section className="bg-white rounded-2xl shadow p-5 space-y-4">
        <h2 className="text-lg font-medium">Get recommendation</h2>
        <div className="flex items-center gap-3">
          <select className="border rounded px-3 py-2" value={policy} onChange={e => setPolicy(e.target.value)}>
            <option value="baseline">Baseline</option>
          </select>
          <button onClick={fetchRecs} disabled={loading} className="px-4 py-2 rounded bg-emerald-600 text-white">
            {loading ? "Thinking..." : "Get next topics"}
          </button>
        </div>
        {recs.length > 0 && (
          <div>
            <h3 className="font-medium mb-2">Recommended next topics</h3>
            <ul className="list-disc pl-6">
              {recs.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}
