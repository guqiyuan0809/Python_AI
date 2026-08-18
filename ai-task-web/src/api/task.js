import http from '../utils/http'

export function createSession() {
  return http.post('/sessions', {})
}

export function submitAsyncTask(payload) {
  return http.post('/sessions/chat/async', payload)
}

export function getTaskStatus(taskId) {
  return http.get(`/tasks/${taskId}`)
}
