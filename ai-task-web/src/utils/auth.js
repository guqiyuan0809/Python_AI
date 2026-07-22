const TOKEN_STORAGE_KEY = 'ai_task_web_token'

export function getToken() {
  return localStorage.getItem(TOKEN_STORAGE_KEY) || ''
}

export function setToken(token) {
  localStorage.setItem(TOKEN_STORAGE_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_STORAGE_KEY)
}
