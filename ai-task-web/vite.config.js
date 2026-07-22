import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // 浏览器只请求 Vite，同源转发到 Java，避免开发阶段跨域。
      '/python-ai': {
        target: 'http://127.0.0.1:9090',
        changeOrigin: true,
      },
    },
  },
})
