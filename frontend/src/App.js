import React, { useEffect, useState } from 'react';
import { getHealth, getTopics, postAttempt, getRecommendation, listClasses, createClass, deleteClass, listClassStudents, addStudentToClass, removeStudentFromClass, uploadStudentCSV, getClassCurriculum, updateClassCurriculum } from './api';
import ClassView from './components/ClassView';
import StudentView from './components/StudentView';

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
  const [view, setView] = useState('class');
  const [classes, setClasses] = useState([]);
  const [newClassName, setNewClassName] = useState('');
  const [curriculumWeeks, setCurriculumWeeks] = useState(['']); // start with Week 1 only
  const [activeClass, setActiveClass] = useState(null);
  const [students, setStudents] = useState([]);
  const [newStudentId, setNewStudentId] = useState('');
  const [newStudentName, setNewStudentName] = useState('');
  const [csvUploading, setCsvUploading] = useState(false);
  // Curriculum (view + edit) for active class
  const [currentCurriculum, setCurrentCurriculum] = useState([]); // read-only display of weeks
  const [editWeeks, setEditWeeks] = useState(['']); // editable weeks state
  const [savingCurr, setSavingCurr] = useState(false);

  useEffect(() => {
    getHealth().then(setStatus).catch(() => setStatus({ ok: false }));
    getTopics().then((t) => { setTopics(t); if (t?.length) setSelTopic(t[0].topic); });
    // Load classes
    listClasses().then(setClasses).catch(() => { });
  }, []);

  const openClass = async (c) => {
    setActiveClass(c);
    setView('students');
    try { setStudents(await listClassStudents(c.id)); } catch { }
    // Load curriculum for this class
    try {
      const cur = await getClassCurriculum(c.id);
      const weeks = (cur?.weeks || []).slice(0, 25);
      setCurrentCurriculum(weeks);
      // Ensure at least Week 1 slot for editing
      setEditWeeks(weeks.length > 0 ? weeks : ['']);
    } catch {
      setCurrentCurriculum([]);
      setEditWeeks(['']);
    }
  };

  const handleCreateClass = async () => {
    if (!newClassName.trim()) return;
    try {
      // Build curriculum: include only weeks where a topic was selected
      // Require Week 1 selection; allow empty for additional weeks
      if (!curriculumWeeks[0] || !curriculumWeeks[0].trim()) { alert('Please select a topic for Week 1'); return; }
      const weeks = curriculumWeeks.map(w => (w && w.trim()) ? w.trim() : '').slice(0, 25);
      const c = await createClass(newClassName.trim(), weeks);
      setClasses([...(classes || []), { id: c.id, name: c.name, student_count: 0 }]);
      setNewClassName('');
      setCurriculumWeeks(['']);
    } catch (e) { alert('Failed to create class'); }
  };

  const handleDeleteClass = async (cls) => {
    if (!window.confirm(`Delete class "${cls.name}"?`)) return;
    try { await deleteClass(cls.id); setClasses((classes || []).filter(x => x.id !== cls.id)); if (activeClass?.id === cls.id) { setActiveClass(null); setStudents([]); setView('class'); } } catch { }
  };

  // Curriculum editing handlers
  const handleCurriculumAddWeek = () => {
    if (editWeeks.length < 25) setEditWeeks([...editWeeks, '']);
  };
  const handleCurriculumRemoveLast = () => {
    if (editWeeks.length > 1) setEditWeeks(editWeeks.slice(0, -1));
  };
  const handleCurriculumSave = async () => {
    if (!activeClass) return;
    // Require Week 1
    if (!editWeeks[0] || !editWeeks[0].trim()) { alert('Please select a topic for Week 1'); return; }
    try {
      setSavingCurr(true);
      const weeks = editWeeks.map(w => (w && w.trim()) ? w.trim() : '').slice(0, 25);
      await updateClassCurriculum(activeClass.id, weeks);
      setCurrentCurriculum(weeks);
      // Normalize edit state to saved weeks
      setEditWeeks(weeks.length > 0 ? weeks : ['']);
    } catch (e) {
      alert('Failed to save curriculum');
    } finally {
      setSavingCurr(false);
    }
  };
  const handleCurriculumCancel = () => {
    // Restore edit form to current curriculum (or Week 1 blank)
    const weeks = (currentCurriculum || []).slice(0, 25);
    setEditWeeks(weeks.length > 0 ? weeks : ['']);
  };

  const handleAddStudent = async () => {
    if (!activeClass) return; if (!newStudentId.trim()) return;
    try { await addStudentToClass(activeClass.id, newStudentId.trim(), newStudentName.trim() || undefined); setStudents([...(students || []), { id: Date.now(), student_id: newStudentId.trim(), name: newStudentName.trim(), class_id: activeClass.id }]); setNewStudentId(''); setNewStudentName(''); } catch { alert('Failed to add student'); }
  };

  const handleRemoveStudent = async (s) => {
    if (!activeClass) return;
    try { await removeStudentFromClass(activeClass.id, s.student_id); setStudents((students || []).filter(x => x.student_id !== s.student_id)); } catch { }
  };

  const handleUploadCSV = async (s, file) => {
    if (!file) return; setCsvUploading(true);
    try { const res = await uploadStudentCSV(s.student_id, file); alert(`Uploaded ${res.inserted} rows`); }
    catch { alert('Upload failed'); }
    finally { setCsvUploading(false); }
  };

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
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Adaptive Quiz Teacher</h1>
        <nav className="space-x-2">
          <button onClick={() => setView('class')} className={`px-3 py-1 rounded ${view === 'class' ? 'bg-blue-600 text-white' : 'border'}`}>Class View</button>
          <button onClick={() => setView('students')} className={`px-3 py-1 rounded ${view === 'students' ? 'bg-blue-600 text-white' : 'border'}`}>Student View</button>
        </nav>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <main className="col-span-2 space-y-6">
          {view === 'class' && (
            <div className="bg-white rounded-2xl shadow p-6 space-y-4">
              <h3 className="text-xl font-semibold">Your Classes</h3>
              <div className="space-y-3">
                <div className="flex gap-2">
                  <input className="border rounded px-3 py-2 flex-1" placeholder="New class name" value={newClassName} onChange={e => setNewClassName(e.target.value)} />
                  <button className="px-3 py-2 rounded bg-blue-600 text-white" onClick={handleCreateClass}>Add</button>
                </div>
                <div className="border rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm font-medium">Set weekly topics (Week 1 required; add weeks as needed)</div>
                    <div className="space-x-2">
                      <button type="button" className="px-2 py-1 text-sm rounded border"
                        onClick={() => {
                          if (curriculumWeeks.length < 25) setCurriculumWeeks([...curriculumWeeks, '']);
                        }}>+ Week</button>
                      <button type="button" className="px-2 py-1 text-sm rounded border"
                        onClick={() => {
                          if (curriculumWeeks.length > 1) setCurriculumWeeks(curriculumWeeks.slice(0, -1));
                        }}>Remove last</button>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                    {curriculumWeeks.map((val, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <label className="w-20 text-xs text-gray-600">Week {idx + 1}</label>
                        <select className={`border rounded px-2 py-1 flex-1 ${idx === 0 && (!val || !val.trim()) ? 'border-red-500' : ''}`}
                          value={val}
                          onChange={e => {
                            const copy = [...curriculumWeeks];
                            copy[idx] = e.target.value;
                            setCurriculumWeeks(copy);
                          }}>
                          <option value="">-- Select Topic --</option>
                          {topics.map(t => (
                            <option key={t.topic} value={t.topic}>{t.topic}</option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {(classes || []).map(c => (
                  <div key={c.id} className="border rounded p-4 bg-gray-50 flex items-center justify-between">
                    <div>
                      <div className="font-medium">{c.name}</div>
                      <div className="text-xs text-gray-500">{c.student_count || 0} students</div>
                    </div>
                    <div className="space-x-2">
                      <button className="px-3 py-1 bg-blue-600 text-white rounded" onClick={() => openClass(c)}>View</button>
                      <button className="px-3 py-1 border rounded" onClick={() => handleDeleteClass(c)}>Delete</button>
                    </div>
                  </div>
                ))}
                {(!classes || classes.length === 0) && <div className="text-sm text-gray-600">No classes yet.</div>}
              </div>
            </div>
          )}

          {view === 'students' && (
            <div className="bg-white rounded-2xl shadow p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold">{activeClass?.name || 'Class'}</h3>
                  <div className="text-sm text-gray-600 flex flex-col md:flex-row md:items-center gap-2">
                    <span>Manage students and upload results CSV per student.</span>
                    <a href="/sample_student_results.csv" download className="text-blue-600 underline">Download sample CSV</a>
                  </div>
                </div>
                <div className="space-x-2">
                  <button onClick={() => setView('class')} className="px-3 py-1 rounded border">Back</button>
                </div>
              </div>

              {/* Current Curriculum */}
              <div className="border rounded p-4 bg-gray-50 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Current Curriculum</h4>
                </div>
                {currentCurriculum && currentCurriculum.filter(w => w && w.trim()).length > 0 ? (
                  <ul className="text-sm list-disc pl-5">
                    {currentCurriculum.map((w, i) => (
                      w && w.trim() ? <li key={i}>Week {i + 1}: {w}</li> : null
                    ))}
                  </ul>
                ) : (
                  <div className="text-sm text-gray-600">No curriculum set yet.</div>
                )}
              </div>

              {/* Edit Curriculum */}
              <div className="border rounded p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Edit Curriculum</h4>
                  <div className="space-x-2">
                    <button type="button" className="px-2 py-1 text-sm rounded border" onClick={handleCurriculumAddWeek}>+ Week</button>
                    <button type="button" className="px-2 py-1 text-sm rounded border" onClick={handleCurriculumRemoveLast}>Remove last</button>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {editWeeks.map((val, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <label className="w-20 text-xs text-gray-600">Week {idx + 1}</label>
                      <select className={`border rounded px-2 py-1 flex-1 ${idx === 0 && (!val || !val.trim()) ? 'border-red-500' : ''}`}
                        value={val}
                        onChange={e => {
                          const copy = [...editWeeks];
                          copy[idx] = e.target.value;
                          setEditWeeks(copy);
                        }}>
                        <option value="">-- Select Topic --</option>
                        {topics.map(t => (
                          <option key={t.topic} value={t.topic}>{t.topic}</option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
                <div className="flex gap-2">
                  {(() => {
                    const norm = (arr) => JSON.stringify((arr || []).map(w => (w && w.trim()) ? w.trim() : ''));
                    const changed = norm(editWeeks) !== norm(currentCurriculum);
                    const invalid = !editWeeks[0] || !editWeeks[0].trim();
                    return (
                      <>
                        <button onClick={handleCurriculumSave} disabled={!changed || invalid || savingCurr} className={`px-3 py-2 rounded ${(!changed || invalid || savingCurr) ? 'bg-gray-300 text-gray-600' : 'bg-emerald-600 text-white'}`}>{savingCurr ? 'Saving...' : 'Save'}</button>
                        <button onClick={handleCurriculumCancel} className="px-3 py-2 rounded border">Cancel</button>
                      </>
                    );
                  })()}
                </div>
              </div>

              <div className="flex gap-2">
                <input className="border rounded px-3 py-2" placeholder="Student ID" value={newStudentId} onChange={e => setNewStudentId(e.target.value)} />
                <input className="border rounded px-3 py-2 flex-1" placeholder="Name (optional)" value={newStudentName} onChange={e => setNewStudentName(e.target.value)} />
                <button onClick={handleAddStudent} className="px-3 py-2 rounded bg-blue-600 text-white">Add Student</button>
              </div>

              <div className="space-y-2">
                {(students || []).map(s => (
                  <div key={s.student_id} className="flex items-center justify-between border rounded p-3 bg-white">
                    <div>
                      <div className="font-medium">{s.student_id}{s.name ? ` - ${s.name}` : ''}</div>
                      <div className="text-xs text-gray-500">Class: {activeClass?.name}</div>
                    </div>
                    <div className="space-x-2">
                      <label className="px-3 py-1 border rounded cursor-pointer">
                        <input type="file" accept=".csv" className="hidden" onChange={e => handleUploadCSV(s, e.target.files[0])} />
                        {csvUploading ? 'Uploading...' : 'Upload CSV'}
                      </label>
                      <button className="px-3 py-1 border rounded" onClick={async () => {
                        try {
                          const resp = await fetch('/sample_student_results.csv');
                          const blob = await resp.blob();
                          const file = new File([blob], 'sample_student_results.csv', { type: 'text/csv' });
                          await handleUploadCSV(s, file);
                        } catch { }
                      }}>Demo Upload</button>
                      <button className="px-3 py-1 border rounded" onClick={() => handleRemoveStudent(s)}>Remove</button>
                    </div>
                  </div>
                ))}
                {(!students || students.length === 0) && <div className="text-sm text-gray-600">No students yet.</div>}
              </div>
            </div>
          )}
        </main>

        <aside className="col-span-1">
          <div className="bg-white rounded-2xl shadow p-5 space-y-4">
            <header className="flex items-center justify-between">
              <h2 className="text-lg font-medium">Control Panel</h2>
              <span className={"text-sm " + (status?.ok ? "text-green-600" : "text-red-600")}>
                API {status?.ok ? "connected" : "offline"}
              </span>
            </header>
            <h4 className="text-sm font-medium">Log an attempt</h4>
            <div className="grid grid-cols-1 gap-2">
              <input className="border rounded px-3 py-2" value={studentId} onChange={e => setStudentId(e.target.value)} placeholder="Student ID" />
              <select className="border rounded px-3 py-2" value={selTopic} onChange={e => setSelTopic(e.target.value)}>
                {topics.map(t => <option key={t.topic} value={t.topic}>{t.topic}</option>)}
              </select>
              <div className="flex gap-2">
                <select className="border rounded px-3 py-2 flex-1" value={difficulty} onChange={e => setDifficulty(Number(e.target.value))}>
                  <option value={1}>Difficulty 1</option>
                  <option value={2}>Difficulty 2</option>
                  <option value={3}>Difficulty 3</option>
                </select>
                <select className="border rounded px-3 py-2" value={correct} onChange={e => setCorrect(Number(e.target.value))}>
                  <option value={1}>Correct</option>
                  <option value={0}>Incorrect</option>
                </select>
              </div>
              <button onClick={submitAttempt} className="px-4 py-2 rounded bg-blue-600 text-white">Save</button>
              {msg && <div className="text-xs text-gray-600">{msg}</div>}
            </div>

            <hr />

            <h4 className="text-sm font-medium">Recommendations</h4>
            <div className="flex gap-2">
              <select className="border rounded px-3 py-2 flex-1" value={policy} onChange={e => setPolicy(e.target.value)}>
                <option value="baseline">Baseline</option>
                <option value="logreg">LogReg</option>
                <option value="cf">CF</option>
                <option value="hybrid">Hybrid</option>
              </select>
              <button onClick={fetchRecs} disabled={loading} className="px-3 py-2 rounded bg-emerald-600 text-white">{loading ? '...' : 'Get'}</button>
            </div>
            {recs.length > 0 && (
              <ul className="mt-2 list-disc pl-5 text-sm">
                {recs.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
