import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

export async function getHealth() { return axios.get(`${API_BASE}/health`).then(r => r.data); }
export async function getTopics() { return axios.get(`${API_BASE}/topics`).then(r => r.data); }
export async function postAttempt(payload) { return axios.post(`${API_BASE}/attempt`, payload).then(r => r.data); }
export async function getRecommendation(studentId, k = 3, policy = 'baseline') {
  return axios.get(`${API_BASE}/recommendation`, { params: { student_id: studentId, k, policy } }).then(r => r.data);
}
