import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

export async function getHealth() { return axios.get(`${API_BASE}/health`).then(r => r.data); }
export async function getTopics() { return axios.get(`${API_BASE}/topics`).then(r => r.data); }
export async function postAttempt(payload) { return axios.post(`${API_BASE}/attempt`, payload).then(r => r.data); }
export async function getRecommendation(studentId, k = 3, policy = 'baseline') {
  return axios.get(`${API_BASE}/recommendation`, { params: { student_id: studentId, k, policy } }).then(r => r.data);
}

// Classes & Students
export async function listClasses() { return axios.get(`${API_BASE}/classes`).then(r => r.data); }
export async function createClass(name) { return axios.post(`${API_BASE}/classes`, { name }).then(r => r.data); }
export async function deleteClass(id) { return axios.delete(`${API_BASE}/classes/${id}`).then(r => r.data); }
export async function listClassStudents(classId) { return axios.get(`${API_BASE}/classes/${classId}/students`).then(r => r.data); }
export async function addStudentToClass(classId, student_id, name) { return axios.post(`${API_BASE}/classes/${classId}/students`, { student_id, name }).then(r => r.data); }
export async function removeStudentFromClass(classId, student_id) { return axios.delete(`${API_BASE}/classes/${classId}/students/${student_id}`).then(r => r.data); }
export async function uploadStudentCSV(student_id, file) {
  const form = new FormData();
  form.append('file', file);
  return axios.post(`${API_BASE}/students/${encodeURIComponent(student_id)}/upload_csv`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
}
