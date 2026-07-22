import axios from 'axios'
import { getToken } from './auth'

// 统一从当前前端域名访问 Java，开发环境由 Vite 代理到 9090 端口。
const http = axios.create({
  baseURL: '/python-ai',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

http.interceptors.request.use((config) => {
  // 每次请求生成 trace_id，方便 Java、Python 两侧串联日志。
  config.headers['X-Trace-Id'] = crypto.randomUUID().replaceAll('-', '')
  // 与公园旧前端保持一致，Java 鉴权层从 token 请求头读取当前登录凭证。
  config.headers.fSource = 2
  const token = getToken()
  if (token) {
    config.headers.token = token
  }
  return config
})

http.interceptors.response.use(
  (response) => {
    const body = response.data
    // Java 的统一响应成功码是 200，业务失败也可能以 HTTP 200 返回。
    if (body?.status !== 200) {
      return Promise.reject(new Error(body?.message || '请求失败'))
    }
    return body.data
  },
  (error) => {
    const message = error.response?.data?.message || error.message || '网络请求失败'
    return Promise.reject(new Error(message))
  },
)

export default http
