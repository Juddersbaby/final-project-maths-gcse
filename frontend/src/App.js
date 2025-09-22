import React, { useEffect, useState } from 'react';
import { getHealth, getTopics, postAttempt, getRecommendation, listClasses, createClass, deleteClass, listClassStudents, addStudentToClass, removeStudentFromClass, uploadStudentCSV, getClassCurriculum, updateClassCurriculum, listPapers, listQuestions, ingestPaper, listStudentAttempts } from './api';
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
  // All students (for Control Panel dropdown)
  const [allStudents, setAllStudents] = useState([]);
  const [allStudentsLoading, setAllStudentsLoading] = useState(false);
  const [studentFilter, setStudentFilter] = useState('');
  // Papers & ingestion
  const [papers, setPapers] = useState([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [ingestUrl, setIngestUrl] = useState('');
  const [ingesting, setIngesting] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState(null);
  const [paperQuestions, setPaperQuestions] = useState([]);
  // Student detail
  const [detailStudent, setDetailStudent] = useState(null);
  const [detailAttempts, setDetailAttempts] = useState([]);
  const [detailRecs, setDetailRecs] = useState([]);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    getHealth().then(setStatus).catch(() => setStatus({ ok: false }));
    getTopics().then((t) => { setTopics(t); if (t?.length) setSelTopic(t[0].topic); });
    // Load classes
    listClasses().then(setClasses).catch(() => { });
    // Load papers (Edexcel, all tiers)
    (async () => {
      try {
        setPapersLoading(true);
        const p = await listPapers('Edexcel');
        setPapers(p || []);
      } catch { setPapers([]); } finally { setPapersLoading(false); }
    })();
  }, []);

  // When classes change, load all students across classes for the Control Panel
  useEffect(() => {
    const loadAll = async () => {
      try {
        setAllStudentsLoading(true);
        const cls = classes || [];
        const lists = await Promise.all(cls.map(async (c) => {
          try {
            const st = await listClassStudents(c.id);
            return (st || []).map(s => ({ ...s, class_name: c.name }));
          } catch { return []; }
        }));
        setAllStudents(lists.flat());
      } finally {
        setAllStudentsLoading(false);
      }
    };
    if (classes && classes.length > 0) {
      loadAll();
    } else {
      setAllStudents([]);
    }
  }, [classes]);

  // Default the selected student to the first available if current selection is not in the list
  useEffect(() => {
    if (!allStudents || allStudents.length === 0) return;
    // Only set a default if none is selected yet; don't override an existing choice
    if (!studentId) {
      setStudentId(allStudents[0].student_id);
    }
  }, [allStudents, studentId]);

  const filteredStudents = React.useMemo(() => {
    const q = (studentFilter || '').toLowerCase();
    if (!q) return allStudents || [];
    return (allStudents || []).filter(s => (
      `${s.student_id} ${s.name || ''} ${s.class_name || ''}`.toLowerCase().includes(q)
    ));
  }, [allStudents, studentFilter]);

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

  // Papers & ingestion handlers
  const handleIngest = async () => {
    if (!ingestUrl.trim()) return;
    try {
      setIngesting(true);
      const res = await ingestPaper(ingestUrl.trim());
      alert(`Ingested Paper ${res.paper_no} (${res.series}, ${res.tier}) with ${res.inserted} questions.`);
      setIngestUrl('');
      // refresh papers
      try { setPapersLoading(true); const p = await listPapers('Edexcel'); setPapers(p || []); } finally { setPapersLoading(false); }
    } catch (e) {
      alert('Failed to ingest paper');
    } finally {
      setIngesting(false);
    }
  };
  const toggleQuestions = async (paper) => {
    if (selectedPaperId === paper.id) {
      setSelectedPaperId(null); setPaperQuestions([]); return;
    }
    setSelectedPaperId(paper.id); setPaperQuestions([]);
    try {
      const qs = await listQuestions(paper.id);
      setPaperQuestions(qs || []);
    } catch { setPaperQuestions([]); }
  };

  const handleAddStudent = async () => {
    if (!activeClass) return; if (!newStudentId.trim()) return;
    try {
      await addStudentToClass(activeClass.id, newStudentId.trim(), newStudentName.trim() || undefined);
      const added = { id: Date.now(), student_id: newStudentId.trim(), name: newStudentName.trim(), class_id: activeClass.id };
      setStudents([...(students || []), added]);
      // Update all-students list with class name
      setAllStudents(prev => [{ ...added, class_name: activeClass?.name }, ...(prev || [])]);
      setNewStudentId(''); setNewStudentName('');
    } catch { alert('Failed to add student'); }
  };

  const handleRemoveStudent = async (s) => {
    if (!activeClass) return;
    try {
      await removeStudentFromClass(activeClass.id, s.student_id);
      setStudents((students || []).filter(x => x.student_id !== s.student_id));
      setAllStudents((allStudents || []).filter(x => x.student_id !== s.student_id));
    } catch { }
  };

  const refreshStudentDetail = async (studentIdParam) => {
    const sid = studentIdParam || detailStudent?.student_id;
    if (!sid) return;
    setDetailAttempts([]); setDetailRecs([]);
    try {
      setDetailLoading(true);
      const [atts, recs] = await Promise.all([
        listStudentAttempts(sid, 100),
        getRecommendation(sid, 3, policy)
      ]);
      setDetailAttempts(atts || []);
      setDetailRecs(recs?.next_topics || []);
    } catch {
      setDetailAttempts([]); setDetailRecs([]);
    } finally { setDetailLoading(false); }
  };

  const openStudentDetail = async (s) => {
    setDetailStudent(s);
    // Also bind Control Panel selection to this student so new attempts save to them
    setStudentId(s.student_id);
    setView('studentDetail');
    await refreshStudentDetail(s.student_id);
  };

  const handleUploadCSV = async (s, file) => {
    if (!file) return; setCsvUploading(true);
    try { const res = await uploadStudentCSV(s.student_id, file); alert(`Uploaded ${res.inserted} rows`); }
    catch { alert('Upload failed'); }
    finally {
      setCsvUploading(false);
      // If we're in the detail view for this student, refresh results/recs
      if (detailStudent && s.student_id === detailStudent.student_id) {
        refreshStudentDetail(detailStudent.student_id);
      }
    }
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

                {/* Papers ingestion */}
                <div className="border rounded p-3 space-y-2">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm font-medium">Ingest Past Paper (MathsGenie URL)</div>
                  </div>
                  <div className="flex gap-2">
                    <input className="border rounded px-3 py-2 flex-1" placeholder="https://www.mathsgenie.co.uk/papers/1fnov2023.pdf" value={ingestUrl} onChange={e => setIngestUrl(e.target.value)} />
                    <button className="px-3 py-2 rounded bg-emerald-600 text-white" onClick={handleIngest} disabled={ingesting}>{ingesting ? 'Ingesting...' : 'Ingest'}</button>
                  </div>
                  <div className="mt-2">
                    <div className="text-sm font-medium mb-1">Papers {papersLoading && <span className="text-gray-500">(loading...)</span>}</div>
                    <div className="space-y-2 max-h-64 overflow-auto">
                      {(papers || []).map(p => (
                        <div key={p.id} className="border rounded p-2 bg-white">
                          <div className="flex items-center justify-between">
                            <div className="text-sm">
                              <span className="font-medium">{p.series}</span> — {p.tier} — Paper {p.paper_no} {p.calculator ? '(Calc)' : ''}
                            </div>
                            <div className="space-x-2">
                              <a href={p.pdf_url} target="_blank" rel="noreferrer" className="text-blue-600 underline text-sm">Open PDF</a>
                              {p.markscheme_url ? (
                                <a href={p.markscheme_url} target="_blank" rel="noreferrer" className="text-blue-600 underline text-sm">Mark Scheme</a>
                              ) : null}
                              <button className="px-2 py-1 text-xs rounded border" onClick={() => toggleQuestions(p)}>{selectedPaperId === p.id ? 'Hide Questions' : 'Show Questions'}</button>
                            </div>
                          </div>
                          {selectedPaperId === p.id && (
                            <div className="mt-2 text-xs">
                              {(paperQuestions || []).length === 0 ? (
                                <div className="text-gray-600">No questions or failed to load.</div>
                              ) : (
                                <ul className="list-disc pl-5 space-y-1">
                                  {paperQuestions.map(q => (
                                    <li key={q.id}>
                                      Q{q.qno} ({q.marks} marks) — {q.topic || 'Unknown'} —
                                      {' '}<a className="text-blue-600 underline" href={`${p.pdf_url}#page=${q.page_start || 1}`} target="_blank" rel="noreferrer">open</a>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                      {(!papers || papers.length === 0) && <div className="text-sm text-gray-600">No papers yet.</div>}
                    </div>
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
                      <button className="px-3 py-1 bg-blue-600 text-white rounded" onClick={() => openStudentDetail(s)}>View</button>
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

          {view === 'studentDetail' && (
            <div className="bg-white rounded-2xl shadow p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold">{detailStudent?.student_id}{detailStudent?.name ? ` - ${detailStudent.name}` : ''}</h3>
                  <div className="text-sm text-gray-600">Class: {activeClass?.name}</div>
                </div>
                <div className="space-x-2">
                  <button onClick={() => refreshStudentDetail()} className="px-3 py-1 rounded border">Refresh</button>
                  <label className="px-3 py-1 border rounded cursor-pointer">
                    <input type="file" accept=".csv" className="hidden" onChange={e => detailStudent && handleUploadCSV({ student_id: detailStudent.student_id }, e.target.files[0])} />
                    Upload CSV
                  </label>
                  <button className="px-3 py-1 border rounded" onClick={async () => {
                    try {
                      const resp = await fetch('/sample_student_results.csv');
                      const blob = await resp.blob();
                      const file = new File([blob], 'sample_student_results.csv', { type: 'text/csv' });
                      if (detailStudent) await handleUploadCSV({ student_id: detailStudent.student_id }, file);
                    } catch {}
                  }}>Demo Upload</button>
                  <button onClick={() => setView('students')} className="px-3 py-1 rounded border">Back</button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border rounded p-4">
                  <h4 className="font-medium mb-2">Results</h4>
                  {detailLoading && <div className="text-sm text-gray-600">Loading...</div>}
                  {(!detailLoading && (!detailAttempts || detailAttempts.length === 0)) && <div className="text-sm text-gray-600">No attempts yet.</div>}
                  {detailAttempts && detailAttempts.length > 0 && (
                    <ul className="text-sm divide-y">
                      {detailAttempts.map((a, i) => (
                        <li key={i} className="py-1 flex items-center justify-between">
                          <span>{a.ts?.slice(0, 19).replace('T', ' ')} — {a.topic} — Diff {a.difficulty}</span>
                          <span className={a.correct ? 'text-emerald-600' : 'text-red-600'}>{a.correct ? 'Correct' : 'Incorrect'}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="border rounded p-4">
                  <h4 className="font-medium mb-2">Recommendations</h4>
                  {detailLoading && <div className="text-sm text-gray-600">Loading...</div>}
                  {(!detailLoading && (!detailRecs || detailRecs.length === 0)) && <div className="text-sm text-gray-600">No recommendations.</div>}
                  {detailRecs && detailRecs.length > 0 && (
                    <ul className="list-disc pl-5 text-sm">
                      {detailRecs.map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  )}
                </div>
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
              <input className="border rounded px-3 py-2" placeholder="Filter students (ID, name, class)" value={studentFilter} onChange={e => setStudentFilter(e.target.value)} />
              <select className="border rounded px-3 py-2"
                value={studentId}
                onChange={e => setStudentId(e.target.value)}>
                {allStudentsLoading && <option>Loading students...</option>}
                {(!allStudentsLoading && (!allStudents || allStudents.length === 0)) && <option value="">No students available</option>}
                {(!allStudentsLoading && allStudents && allStudents.length > 0 && filteredStudents.length === 0) && <option value="">No matches</option>}
                {filteredStudents.map(s => (
                  <option key={s.student_id} value={s.student_id}>
                    {s.student_id}{s.name ? ` - ${s.name}` : ''}{s.class_name ? ` (Class: ${s.class_name})` : ''}
                  </option>
                ))}
              </select>
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
