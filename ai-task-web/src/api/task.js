import http from '../utils/http'

export function submitAsyncTask(payload) {
  return http.post('/sessions/chat/async', payload)
}

export function getTaskStatus(taskId) {
  return http.get(`/tasks/${taskId}`)
}
